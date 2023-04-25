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
import utils
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

# 0. prepare args and logger
parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="merge.json")
parser.add_argument("--output_path", type=str, default="lora-Vicuna")
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--micro_batch", type=int, default=4)
parser.add_argument("--total_batch", type=int, default=128)
parser.add_argument("--log_steps", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--warmup_steps", type=int, default=200)
parser.add_argument("--test_size", type=int, default=200)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
args = parser.parse_args()

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"
MICRO_BATCH_SIZE = args.micro_batch  # this could actually be 5 but i like powers of 2
BATCH_SIZE = args.total_batch
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3 
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 768  
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.test_size  # 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
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

logger = utils.set_console_logger(__name__)
# 1. load dataset
logger.info(f'>>> processing data from {DATA_PATH}')
train_tokenizer = LlamaTokenizer.from_pretrained(args.model_path, add_eos_token=True)
# unk. we want this to be different from the eos token
train_tokenizer.pad_token_id = 0  
# cannot use eos in generation!
# tokenizer.padding_side = "left"  # Allow batched inference
test_tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
data = load_dataset('json', data_files=DATA_PATH)
# data = utils.from_jsonl(DATA_PATH)
# data = Dataset.from_json(DATA_PATH)
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
CHAT_DICT = {
    'prompt1': (
        "The following is a conversation between an AI assistant called Assistant and a human user called User. "
        "The assistant is intelligent, knowledgeable and polite to answer questions of user.\n\n"
    ),
    'prompt1.5': 'System:{context}\n\n',
    'prompt2': "User:{input}\n\nAssistant:{output}\n\n",
    'prompt3': "User:{input}\n\nAssistant:"
}

def generate_and_tokenize_prompt_0(data_point, stage='train'):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response
    logger.debug(data_point)
    user_prompt = CHAT_DICT['prompt1']
    lens = len(data_point)
    for i in range(lens-1):
        user_prompt += CHAT_DICT['prompt2'].format_map(data_point[i])
    user_prompt += CHAT_DICT['prompt3'].format_map(data_point[-1])
    logger.debug(user_prompt)
    if stage == 'train':
        len_user_prompt_tokens = (len(train_tokenizer(
            user_prompt,
            truncation=True,
            max_length=CUTOFF_LEN + 1,
        )["input_ids"])- 1)  # no eos token
        full_tokens = train_tokenizer(
            user_prompt + data_point["output"],
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )["input_ids"][:-1]
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }
    else:
        inputs = test_tokenizer(user_prompt, return_tensors="pt")["input_ids"]
        return inputs

def generate_and_tokenize_prompt(data_point, stage='train'):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    logger.debug(data_point)
    user_prompt = CHAT_DICT['prompt1']
    lens = len(data_point['input'])
    for i in range(lens-1):
        user_prompt += CHAT_DICT['prompt2'].format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
    user_prompt += CHAT_DICT['prompt3'].format_map({'input':data_point['input'][-1]})
    logger.debug(user_prompt)
    if stage == 'train':
        len_user_prompt_tokens = (len(train_tokenizer(
            user_prompt,
            truncation=True,
            max_length=CUTOFF_LEN + 1,
        )["input_ids"])- 1)  # no eos token
        full_tokens = train_tokenizer(
            user_prompt + data_point["output"][-1],
            truncation=True,
            padding=False,
            max_length=CUTOFF_LEN + 1,
        )["input_ids"][:-1]
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }
    else:
        inputs = test_tokenizer(user_prompt, return_tensors="pt")["input_ids"]
        return inputs

# speedup dataset processing by multi-process
num_proc =(os.cpu_count()//2+1)
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=num_proc)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt, num_proc=num_proc)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=num_proc)
    val_data = None

# 2. load model and checkpoints
logger.info(f'>>> load model from {args.model_path}')
model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map=device_map,
)
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
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
            warnings.warn("epoch {} replace to the base_max_steps {}".format(EPOCHS, base_max_steps))
            EPOCHS = None
            MAX_STEPS = base_max_steps
        else:
            MAX_STEPS = now_max_steps
    assert MAX_STEPS is not None
    checkpoint_name = os.path.join(
        args.resume_from_checkpoint, "pytorch_model.bin"
    )
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.bin"
        )
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn("The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = None  # So the trainer won't try loading its state
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        logger.info(f'>>> load lora from {checkpoint_name}')
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")
else:
    MAX_STEPS = now_max_steps

model.print_trainable_parameters()

# 3. start training
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self.trainer = trainer
        self.generation_config = GenerationConfig(
            temperature=1.0,
            top_p=0.75,
            top_k=40,
            num_beams=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=1024, # max_length=max_new_tokens+input_sequence
            min_new_tokens=1, # min_length=min_new_tokens+input_sequence
            bad_words_ids=test_tokenizer(['\n\nUser:','\n\nAssistant:'], add_special_tokens=False).input_ids
        )
        self.repetition_penalty=1.3
        self.logger = utils.set_file_logger('transformers.trainer', trainer.args.output_dir)
        self.test_file = 'sample/test_multi.jsonl'

    def on_train_begin(self, args, state, control, **kwargs):
        # self.test(self.trainer.model, args,state)
        pass

    # save model的时候调用
    def on_save(self, args, state, control, model, **kwargs):
        # self.test(model, args,state)
        pass

    def test(self, model, args, state):
        checkpoint_folder = f"checkpoint-{state.global_step}"
        run_dir = self.trainer._get_output_dir(trial=None)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'gen.txt')
        # no batch 只能测单轮对话
        # self.trainer.model 和 model 是同一个
        model.eval()
        with torch.no_grad():
            test_datas = utils.from_jsonl(self.test_file)
            total = len(test_datas)
            for data in test_datas:
                inputs = generate_and_tokenize_prompt_0(data, 'test')
                len_input = len(inputs[0])
                input_ids = inputs.to(args.device)
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                    repetition_penalty=self.repetition_penalty,
                )
                output = generation_output.sequences[0]
                # data['output'] = test_tokenizer.decode(output).split("Assistant:")[-1].strip()
                data[-1]['gen'] = test_tokenizer.decode(output[len_input:])
                logger.info(data)
            utils.to_jsonl(test_datas, output_file)
        model.train()

    def on_log(self, args, state, control, logs, **kwargs):
        logger.info(logs)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=args.warmup_steps,
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
    data_collator=transformers.DataCollatorForSeq2Seq(train_tokenizer)
)
trainer.add_callback(CustomCallback(trainer))
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

logger.info("\n You can disregard the warning about missing keys \n 下面关于missing keys的warning可以忽略")

try:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(OUTPUT_DIR)
except:
    import sys,pdb,bdb
    type, value, tb = sys.exc_info()
    if type == bdb.BdbQuit:
        exit()
    print(type,value)
    pdb.post_mortem(tb)
