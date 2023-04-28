import sys
import torch
import torch.nn as nn
import transformers
import gradio as gr
import argparse
import warnings
import os
import quant
from gptq import GPTQ
from datautils import get_loaders

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint), strict=False)

    quant.make_quant_attn(model)
    if eval and fused_mlp:
        quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}

        ### Input:
        {input}
        
        ### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Response:"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="decapoda-research/llama-7b-hf",help="llama huggingface model to load")
    parser.add_argument("--quant_path",type=str,default="llama7b-8bit-128g.pt",help="the quantified model path")
    parser.add_argument(
                        "--wbits",
                        type=int,
                        default=4,
                        choices=[2, 3, 4, 8],
                        help="bits to use for quantization; use 8 for evaluating base model.")
    
    parser.add_argument('--text', type=str, default='the mean of life is', help='input text')

    parser.add_argument('--min_length', type=int, default=10, help='The minimum length of the sequence to be generated.')

    parser.add_argument('--max_length', type=int, default=256, help='The maximum length of the sequence to be generated.')

    parser.add_argument('--top_p',
                        type=float,
                        default=0.95,
                        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.')

    parser.add_argument('--temperature', type=float, default=0.1, help='The value used to module the next token probabilities.')
    parser.add_argument('--repetition_penalty',type=float, default=2.0, help='The parameter for repetition penalty. 1.0 means no penalty(0~10)')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--gradio', action='store_true', help='Whether to use gradio to present results.')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = load_quant(args.model_path, args.quant_path, args.wbits, args.groupsize)
    model.to(device)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    #[Way1]: drectly generate
    if not args.gradio:
        input_ids = tokenizer.encode(args.text, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                min_new_tokens=args.min_length,
                max_new_tokens=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )
        print("*"*80)
        print("🦙:", tokenizer.decode([el.item() for el in generated_ids[0]],skip_special_tokens=True))
    #[Way2]: generate through the gradio interface
    else:   
        def evaluate(
            input,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            max_new_tokens=128,
            repetition_penalty=1.0,
            **kwargs,
        ):
            prompt = generate_prompt(input)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                **kwargs,
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=float(repetition_penalty),
                )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s,skip_special_tokens=True)
            return output.split("### Response:")[1].strip()


        gr.Interface(
            fn=evaluate,
            inputs=[
                gr.components.Textbox(
                    lines=2, label="Input", placeholder="Tell me about alpacas."
                ),
                gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
                gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
                gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
                gr.components.Slider(minimum=1, maximum=5, step=1, value=1, label="Beams"),
                gr.components.Slider(
                    minimum=1, maximum=2000, step=1, value=256, label="Max tokens"
                ),
                gr.components.Slider(
                    minimum=0.1, maximum=10.0, step=0.1, value=1.0, label="Repetition Penalty"
                ),
            ],
            outputs=[
                gr.inputs.Textbox(
                    lines=5,
                    label="Output",
                )
            ],
            title="Chinese-Vicuna 中文小羊驼",
            description="中文小羊驼由各种高质量的开源instruction数据集，结合Alpaca-lora的代码训练而来，模型基于开源的llama7B，主要贡献是对应的lora模型。由于代码训练资源要求较小，希望为llama中文lora社区做一份贡献。",
        ).launch(share=True)


if __name__ == '__main__':
    main()
