import os
import tqdm
import joblib
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel
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

# reload the model: no int8, so 14GB is needed
version = 'no.pt' # finetune_0.pt
model_dir = '/home/liang/lzy_tmp/models/chatglm-6b'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
config = LoraConfig(
    peft_type="LORA", 
    task_type="SEQ_2_SEQ_LM", 
    r=32, 
    lora_alpha=32, 
    target_modules=["q", "k", "v"],
    lora_dropout=0.1, 
)

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

if version != 'no.pt':
    # convert it again
    for key, module in model.named_modules():
        if key.endswith('attention'):
            try:
                qkv_layer = QKV_layer(module.query_key_value.in_features, module.query_key_value.out_features) 
                qkv_layer.update(module.query_key_value)
                module.query_key_value = qkv_layer
            except:
                pass
            module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value)

    # load the LoRA checkpoint
    model.load_state_dict(torch.load(f'/{model_dir}/{version}'), strict=False)

model.half().cuda().eval()

# Let's chat!
os.makedirs('outs/chatglm-6b/', exist_ok=True)
with open(f'outs/chatglm-6b/test_{version}.txt','w') as f:
    for text in open('sample/test.jsonl'):
        text = json.loads(text)
        inputs = text['instruction']
        print('Q:', inputs)
        print('Q:', inputs, file=f)
        response, history = model.chat(tokenizer, inputs, history=[])
        print('A:', response)
        print('A:', response, '\n',file=f)
