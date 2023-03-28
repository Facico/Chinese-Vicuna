from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import torch
import sys

def load_model(
    base="decapoda-research/llama-7b-hf",
    finetuned="tloen/alpaca-lora-7b",
    local_path=True,
    LOAD_8BIT=True,
):
    tokenizer = LlamaTokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    lora_bin_path = os.path.join(finetuned, "adapter_model.bin")
    print(finetuned)
    if not os.path.exists(lora_bin_path) and local_path is True:
        pytorch_bin_path = os.path.join(finetuned, "pytorch_model.bin")
        print(pytorch_bin_path)
        if os.path.exists(pytorch_bin_path):
            os.rename(pytorch_bin_path, lora_bin_path)
            warnings.warn("The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'")
        else:
            assert ('Checkpoint is not Found!')
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass
    """model = LlamaForCausalLM.from_pretrained(
        base,
        load_in_8bit=True,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(model, finetuned, device_map={'': 0})"""
    BASE_MODEL = base
    LORA_WEIGHTS = finetuned
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model, tokenizer

