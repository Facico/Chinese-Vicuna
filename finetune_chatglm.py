
### Load Model From huggingface

import os
import tqdm
import joblib
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel
import wandb
import peft
import loralib as lora
from peft import LoraConfig

import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup


"""
extra requirements: 
    pip install icetk
"""
checkpoint = "/model/chatglm-6b"
datafile='datasets/merge.json'
out_dir= 'outs/chatglm-6b'
use_wandb=True

mixed_precision = 'bf16'
accumulate_step = 8
log_interval = 100
Per_GPU_BATCH_SIZE = 2
MAX_LENGTH = 256 # have huge impact on VRAM: 968:1, 256:4
config = LoraConfig(
    peft_type="LORA", 
    r=32,
    lora_alpha=32,
    target_modules=["q", "k", "v"],
    lora_dropout=0.1, 
)
LR = 2e-5
NUM_EPOCHS = 3
warm_up_ratio = 0.1
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
if use_wandb:
    wandb.init(
        project="LoRA",
        name=f"{checkpoint}-{datafile}",
        config=None,
    )
else:
    wandb.init(mode='disabled')

os.makedirs(out_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, 
    trust_remote_code=True,
    device_map=device_map,
)
# BUG: must remove special token '[MASK]'
# del tokenizer.vocab['MASK'] 


### Dataset
EOS_ID = 150005
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

with open(datafile, 'r') as f:
    content = json.load(f)
pairs = []
for line in content:
    if line['input'] == '':
        prompt = PROMPT_DICT['prompt_no_input'].format_map(line)
    else:
        prompt = PROMPT_DICT['prompt_input'].format_map(line)
    completion = line['output']+'</s>'
    if len(prompt) + len(completion) < MAX_LENGTH:
        pairs.append({'prompt':prompt, 'completion':completion})

class AlpacaDataset(Dataset):
    def __init__(self, pairs, tokenizer) -> None:
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer
 
    def __getitem__(self, index):
        if self.pairs[index]['completion'][-4:] == '</s>':
            prompt = self.tokenizer.encode(self.pairs[index]['prompt'])
            completion = self.tokenizer.encode(self.pairs[index]['completion'][:-4], add_special_tokens=False)
            completion += [EOS_ID]
        else:
            prompt = self.tokenizer.encode(self.pairs[index]['prompt'])
            completion = self.tokenizer.encode(self.pairs[index]['completion'], add_special_tokens=False)
        if 150001 not in prompt:
            prompt = self.pairs[index]['prompt'].replace('[MASK]', '//MASK//').replace('[gMASK]', '//gMASK//')
            completion = self.pairs[index]['completion'].replace('[MASK]', '//MASK//').replace('[gMASK]', '//gMASK//')
            prompt = self.tokenizer.encode(prompt)
            completion = self.tokenizer.encode(completion, add_special_tokens=False)
            if 150001 not in prompt:
                import pdb; pdb.set_trace()
        return {'prompt':prompt, 'completion':completion}

    def __len__(self):
        return len(self.pairs)

def collate_fn(batch):
    input_ids = []
    labels = []
    position_ids = []
    device='cuda:0'
    _max_length = max([len(obj['prompt'])+len(obj['completion']) for obj in batch])
    attention_mask = torch.ones((len(batch), _max_length, _max_length), device=device)
    attention_mask.tril_()

    for i, obj in enumerate(batch):
        context_length = obj['prompt'].index(150004)
        attention_mask[i, :, :context_length] = 1

        to_pad = _max_length - len(obj['prompt']) - len(obj['completion'])

        input_ids.append(obj['prompt'] + obj['completion'] + [tokenizer.pad_token_id] * to_pad)

        position_ids.append(torch.stack(
            [torch.arange(0, _max_length, device=device), 
            torch.concat([torch.zeros(context_length - 1, device=device), 
            torch.arange(0, _max_length - context_length + 1, device=device)])]).long()
        )
        labels.append(torch.tensor([-100] * len(obj['prompt']) + obj['completion'] + [-100] * to_pad, device=device).long())
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    return {'input_ids': torch.tensor(input_ids).long(), 
            'attention_mask': attention_mask, 
            'labels': torch.stack(labels),
            'position_ids':torch.stack(position_ids)}

train_dataset = AlpacaDataset(pairs,tokenizer=tokenizer,)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn = collate_fn, shuffle=True, batch_size=Per_GPU_BATCH_SIZE)

# check
for step, batch in enumerate(t:=tqdm.tqdm(train_dataloader)):
    pass

model = AutoModel.from_pretrained(
    checkpoint, 
    trust_remote_code=True,
)
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)
device = accelerator.device


### Insert LoRA to model
class QKV_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(QKV_layer, self).__init__()
        self.linear_q = torch.nn.Linear(in_features, out_features//3)
        self.linear_k = torch.nn.Linear(in_features, out_features//3)
        self.linear_v = torch.nn.Linear(in_features, out_features//3)

    def update(self, target_layer):
        self.linear_q.weight.data = target_layer.weight[:target_layer.out_features//3, :].data
        self.linear_q.bias.data = target_layer.bias[:target_layer.out_features//3].data

        self.linear_k.weight.data = target_layer.weight[target_layer.out_features//3:target_layer.out_features//3*2, :].data
        self.linear_k.bias.data = target_layer.bias[target_layer.out_features//3:target_layer.out_features//3*2].data

        self.linear_v.weight.data = target_layer.weight[target_layer.out_features//3*2:, :].data
        self.linear_v.bias.data = target_layer.bias[target_layer.out_features//3*2:].data
    
    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return torch.concat([q,k,v], dim = -1)

for key, module in model.named_modules():
    if key.endswith('attention'):
        if isinstance(module.query_key_value, peft.tuners.lora.LoraModel):
            module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value.model)
        else:
            # Here we split the query_key_value layer into three linear layer for LoRA. But you can also use merged linear.
            qkv_layer = QKV_layer(module.query_key_value.in_features, module.query_key_value.out_features) 
            qkv_layer.update(module.query_key_value)
            module.query_key_value = qkv_layer
            module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value)

lora.mark_only_lora_as_trainable(model)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
trainable_params = sum([np.prod(p.size()) for p in model_parameters])
non_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
print('trainable_params:{} ({:.2f}%), non_trainable_params:{}'.format(
    trainable_params, trainable_params/non_trainable_params*100,non_trainable_params
))

### Training

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
    num_training_steps=(int(len(train_dataloader) / accumulate_step) * NUM_EPOCHS),
)
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
model.to(device).train()

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for step, batch in enumerate(t:=tqdm.tqdm(train_dataloader)):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss_detach = outputs.loss.detach().cpu().float()
            # t.set_description(f"loss: {loss_detach}")
            t.set_postfix(loss=loss_detach.item())
            total_loss += loss_detach
            loss = outputs.loss

            if accelerator.is_main_process:
                if step % log_interval == 0:
                    wandb.log({
                        'train/loss': loss_detach.item(),
                    })

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        peft_model_id = f"finetune_{epoch}"
        accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), f'{out_dir}/{peft_model_id}.pt')
    