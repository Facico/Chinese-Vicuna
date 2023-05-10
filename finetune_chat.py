from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainerCallback, GenerationConfig
import os
import sys
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import transformers
import argparse
import warnings
from tqdm import tqdm
from functools import partial
import utils
import prompt
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

# 0. prepare args and logger
parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--prompt_type", type=str, default="chat")
parser.add_argument("--data_path", type=str, default="merge.json")
parser.add_argument("--output_path", type=str, default="lora-Vicuna")
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--micro_batch", type=int, default=4)
parser.add_argument("--total_batch", type=int, default=128)
parser.add_argument("--log_steps", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--warmup_ratio", type=float, default=0.05)
parser.add_argument("--test_size", type=int, default=200)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=bool, default=False)
args = parser.parse_args()
if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"
MICRO_BATCH_SIZE = args.micro_batch  # this could actually be 5 but i like powers of 2
BATCH_SIZE = args.total_batch
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3 
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 512  
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.test_size  # 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "down_proj",
    "gate_proj",
    "up_proj",
]
DATA_PATH = args.data_path  # "/home/cciip/private/fanchenghao/dataset/instruction/merge.json"
OUTPUT_DIR = args.output_path  # "lora-Vicuna"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
# we must make sure batch_size and gradient_accumulation_steps not changed for resuming training.
if args.resume_from_checkpoint:
    old_args_path = os.path.join(args.resume_from_checkpoint, 'training_args.bin')
    if os.path.exists(old_args_path):
        old_args = torch.load(old_args_path)
        if MICRO_BATCH_SIZE != old_args.per_device_train_batch_size:
            raise Exception(
                f'current micro batch size {MICRO_BATCH_SIZE} is not equal to the old {old_args.per_device_train_batch_size},'
                ' This will cause the trainer skips wrong epochs or steps.'
                f'please change your micro batch size to {old_args.per_device_train_batch_size}'
                ' or cancel resuming your training'
                )
        if GRADIENT_ACCUMULATION_STEPS != old_args.gradient_accumulation_steps:
            raise Exception(
                f'current total batch {BATCH_SIZE} is not equal to the old {old_args.gradient_accumulation_steps*old_args.per_device_train_batch_size},'
                ' This will cause the trainer skips wrong epochs or steps.'
                f'please change your total batch size to {old_args.gradient_accumulation_steps*old_args.per_device_train_batch_size}'    
                ' or cancel resuming your training'
            )
    else:
        raise Exception(f'{old_args_path} is not exist!')
    # checkpoint = os.path.join(args.resume_from_checkpoint, 'pytorch_model.bin')

logger = utils.set_file_logger(__name__,OUTPUT_DIR)
# 1. load dataset
logger.info(f'>>> processing data from {DATA_PATH}')
logger.info(f'>>> using {args}')

train_tokenizer = LlamaTokenizer.from_pretrained(args.model_path, add_eos_token=True)
# unk. we want this to be different from the eos token
train_tokenizer.pad_token_id = 0  
# cannot use eos in generation!
# tokenizer.padding_side = "left"  # Allow batched inference
test_tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
if args.prompt_type == 'instruct':
    PROMPT = prompt.instruct_prompt(train_tokenizer, CUTOFF_LEN)
elif args.prompt_type == 'chat':
    PROMPT = prompt.chat_prompt(train_tokenizer,CUTOFF_LEN)
else:
    raise Exception('not support')
# check tokenizer
data = load_dataset('json', data_files=DATA_PATH)
import random;start = random.randint(1, 100)
examples = Dataset.from_dict(data['train'][start:start+5]).map(PROMPT.preprocess_train)
for example in examples:
    logger.info(f'>>> using prompt {args.prompt_type}, prompt example:\n { train_tokenizer.decode(example["input_ids"]) }')
    logger.info(f'>>> tokenizer labels: { train_tokenizer.decode([ 0 if l==-100 else l for l in example["labels"]])}')
    logger.info(f'>>> tokenizer example: { example["input_ids"][:10] }...{ example["input_ids"][-10:]}')

# 2. load model and checkpoints
logger.info(f'>>> load model from {args.model_path}')
model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=False,
    device_map=device_map,
    torch_dtype=torch.float16,
).half()
#model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
if args.resume_from_checkpoint:
    checkpoint_name = os.path.join(args.resume_from_checkpoint, "pytorch_model.bin")
    # adapter_model.bin
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            logger.warning("The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = None  # So the trainer won't try loading its state
    # pytorch_model.bin
    if os.path.exists(checkpoint_name):
        logger.info(f'>>> load lora from {checkpoint_name}')
        adapters_weights = torch.load(checkpoint_name)
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        raise Exception(f"Checkpoint {checkpoint_name} not found with resume_from_checkpoint=True!")

trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    num_params = param.numel()
    # if using DS Zero 3 and the weights are initialized empty
    if num_params == 0 and hasattr(param, "ds_numel"):
        num_params = param.ds_numel
    all_param += num_params
    if param.requires_grad:
        trainable_params += num_params
logger.info(f">>> trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# 3. speedup dataset processing by multi-process
num_proc = (os.cpu_count())
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(PROMPT.preprocess_train, num_proc=num_proc)
    val_data = train_val["test"].shuffle().map(PROMPT.preprocess_train, num_proc=num_proc)
else:
    train_data = data["train"].shuffle().map(PROMPT.preprocess_train1, num_proc=num_proc)
    val_data = None
now_max_steps = max((len(data["train"]) - VAL_SET_SIZE) // BATCH_SIZE * EPOCHS, EPOCHS)
if args.resume_from_checkpoint:
    # the trainer will ignore the state max_steps and caculate max_steps based on epochs,
    # so we mannally set the args.max_step to override it. 
    train_state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
    if os.path.exists(train_state_path):
        import json
        base_train_args = json.load(open(train_state_path, 'r'))
        base_max_steps = base_train_args["max_steps"]
        resume_scale = base_max_steps / now_max_steps
        if base_max_steps > now_max_steps:
            logger.warning(f"epoch {EPOCHS}:{MAX_STEPS} replace to the base_max_steps {base_max_steps}")
            EPOCHS = None
            MAX_STEPS = base_max_steps
        else:
            MAX_STEPS = now_max_steps
    assert MAX_STEPS is not None
else:
    MAX_STEPS = now_max_steps

# 4. start training
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self.trainer = trainer
        self.generation_config = GenerationConfig(
            temperature=1.0,
            top_p=0.75,
            top_k=40,
            num_beams=2,
            bos_token_id=train_tokenizer.bos_token_id,
            eos_token_id=train_tokenizer.eos_token_id,
            pad_token_id=train_tokenizer.pad_token_id,
            max_new_tokens=1024, # max_length=max_new_tokens+input_sequence
            min_new_tokens=1, # min_length=min_new_tokens+input_sequence
            bad_words_ids=test_tokenizer(['\n\nUser:','\n\nAssistant:'], add_special_tokens=False).input_ids
        )
        self.repetition_penalty=1.3
        self.logger = utils.set_file_logger('transformers.trainer', trainer.args.output_dir)

    def on_log(self, args, state, control, logs, **kwargs):
        logger.info(logs)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=args.log_steps,
        logging_first_step=True, # convenient
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
    ),
    data_collator=PROMPT.data_collator()
)
trainer.add_callback(CustomCallback(trainer))
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
model.save_pretrained(OUTPUT_DIR)