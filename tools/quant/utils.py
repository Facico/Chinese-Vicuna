import torch
from .modeling_llama import LLaMAForCausalLM


def non_ops(*args, **kwargs):
    pass


def avoid_tensor_modified():
    torch.nn.init.kaiming_uniform_ = non_ops
    torch.nn.init.uniform_ = non_ops
    torch.nn.init.normal_ = non_ops


def get_llama(model, seqlen=1024):
    avoid_tensor_modified()
    model = LLaMAForCausalLM.from_pretrained(model, torch_dtype="auto")
    model.seqlen = seqlen
    return model
