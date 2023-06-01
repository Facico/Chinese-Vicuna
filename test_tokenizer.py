import os
import sys
import torch
import transformers
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

    
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="yahma/llama-7b-hf") #yahma/llama-7b-hf #decapoda-research/llama-7b-hf
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(
    args.model_path, add_eos_token=True
)

test_text = ["Hello, nice to meet you!", "你好很高兴能见到你！"]

for text in test_text:
    input_ids = tokenizer.encode(text)
    print(f"input_ids: {input_ids}")
    decode_text = tokenizer.decode(input_ids)
    print(f"decode_text: {decode_text}")

"""
Correct ==>  yahma/llama-7b-hf + newest Transformers(>=4.28.1):
> !!! Beginning with 1 (bos), ending with 2 (eos) !!!

input_ids: [1, 15043, 29892, 7575, 304, 5870, 366, 29991, 2]
decode_text: <s> Hello, nice to meet you!</s>
input_ids: [1, 29871, 30919, 31076, 232, 193, 139, 30528, 31914, 30815, 235, 170, 132, 30780, 30919, 30584, 2]
decode_text: <s> 你好很高兴能见到你！</s>

Correct ==> decapoda-research/llama-7b-hf + Old Transformers like our version(transformers @ git+https://github.com/huggingface/transformers.git@0dcb46e7a4a9e587ba84ff35778ab4233a184c11)
input_ids: [1, 15043, 29892, 7575, 304, 5870, 366, 29991, 2]
decode_text:  Hello, nice to meet you!
input_ids: [1, 29871, 30919, 31076, 232, 193, 139, 30528, 31914, 30815, 235, 170, 132, 30780, 30919, 30584, 2]
decode_text:  你好很高兴能见到你！

Correct ==> decapoda-research/llama-7b-hf + Old Transformers like our version(transformers @ git+https://github.com/huggingface/transformers.git@0dcb46e7a4a9e587ba84ff35778ab4233a184c11)
input_ids: [1, 15043, 29892, 7575, 304, 5870, 366, 29991, 2]
decode_text:  Hello, nice to meet you!
input_ids: [1, 29871, 30919, 31076, 232, 193, 139, 30528, 31914, 30815, 235, 170, 132, 30780, 30919, 30584, 2]
decode_text:  你好很高兴能见到你！


老版本transformers的问题：代码默认加载tokenizer.model
新版本transformers的修改：新版本默认加载config

decapoda-research：config的bos=2，eos=1（×），tokenizer.model是正确的
yahma：config的bos=1，eos=2，tokenizer.model是正确的
"""