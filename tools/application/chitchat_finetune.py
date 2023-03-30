import os
import sys
import wandb
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# Used for chitchat dataset
# 用于闲聊对话数据

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="datasets/chitchat-1e5.json") # for example: LCCC 
parser.add_argument("--output_path", type=str, default="outs/13B")
parser.add_argument("--model_path", type=str, default="../model/13B_hf")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--test_size", type=int, default=0)
args = parser.parse_args()
# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 24  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2  # we don't always need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 341  # max:341
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.test_size #2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
DATA_PATH = args.data_path #"/home/cciip/private/fanchenghao/dataset/instruction/merge.json"
OUTPUT_DIR = args.output_path #"lora-Vicuna"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
if args.wandb:
    wandb.login(key = '41327ad68395c1a5e5e3827fa5ee97944740250d') # luzhenyi
    wandb.init(
        project="LoRA",
        name=f"{args.model_path}-{args.data_path}",
        config=None,
    )
else:
    wandb.init(mode='disabled')

tokenizer = LlamaTokenizer.from_pretrained(
    args.model_path, add_eos_token=True
)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files=DATA_PATH)

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
    'prompt': (
        "The following is a conversation between an AI assistant called Bot and a human user called User."
        "Bot is is intelligent, knowledgeable, wise and polite.\n\n"
    ),
    'history': (
        "User:\n{input}\n\nBot:{output}\n\n"
    ),
    'input': (
        "### User:\n{input}\n\n### Bot:"
    )
}

def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }
def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = CHAT_DICT['prompt']
    for history in data_point['history']:
        user_prompt+= CHAT_DICT['history'].format_map(history) 
    user_prompt += CHAT_DICT['input'].format_map(data_point)
    len_user_prompt_tokens = (len(tokenizer(
        user_prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
    )["input_ids"])- 1)  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length", # pad到最长
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt,num_proc=12)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt,num_proc=12)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt,num_proc=12)
    val_data = None

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

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
print("\n If there's a warning about missing keys above, please disregard :)")