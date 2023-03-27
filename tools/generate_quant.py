import sys
import torch
from peft import PeftModel
import transformers
import gradio as gr
import argparse
import warnings
import os
from gptq import find_layers, make_quant
from quant.configuration_llama import LLaMAConfig
from quant.modeling_llama import LLaMAForCausalLM
from quant.utils import avoid_tensor_modified
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

def load_quant(model_name, checkpoint, wbits, seqlen=1024, for_infer=True):
    """
    seqlen - seqlen refers to the maximum length of the input sequence that the model can process. The input sequence can be a sequence of words, tokens, or characters, depending on how the model is tokenized. The seqlen parameter is important because it determines the amount of memory that the model requires to process the input sequence. If the input sequence is too long, it may exceed the memory capacity of the model, leading to out-of-memory errors or slower inference times. In order to handle longer sequences, some models use techniques such as attention masking or truncation, which allow the model to process only a portion of the input sequence at a time. The seqlen parameter determines the maximum length of the input sequence that can be processed in a single step. If the input sequence is longer than the seqlen parameter, it may need to be split into multiple segments and processed separately.
    """
    import transformers

    config = LLaMAConfig.from_pretrained(model_name)
    avoid_tensor_modified()

    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LLaMAForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if for_infer:
        model = model.eval()
    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits)

    print(f"⌛️ Loading model from {checkpoint}...")
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = seqlen
    print(f"✅ Model from {checkpoint} is loaded successfully.")

    return model


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
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


def evaluate(
    tokenizer,
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    repetition_penalty=1.0,
    **kwargs,
):
    prompt = generate_prompt(input)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # input_ids = inputs["input_ids"]
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
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="decapoda-research/llama-7b-hf",help="llama model to load")
    parser.add_argument("--quant_path",type=str,required=True,help="the quantified model path")
    parser.add_argument(
        "--wbits",
        type=int,
        default=2,
        choices=[2, 3, 4, 8],
        help="bits to use for quantization; use 8 for evaluating base model."
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = load_quant(args.model_path, args.quant_path, args.wbits)
    model.to(device)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

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
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()


    # gr.Interface(
    #     fn=evaluate,
    #     inputs=[
    #         gr.components.Textbox(
    #             lines=2, label="Input", placeholder="Tell me about alpacas."
    #         ),
    #         gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
    #         gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
    #         gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
    #         gr.components.Slider(minimum=1, maximum=5, step=1, value=1, label="Beams"),
    #         gr.components.Slider(
    #             minimum=1, maximum=2000, step=1, value=256, label="Max tokens"
    #         ),
    #         gr.components.Slider(
    #             minimum=0.1, maximum=10.0, step=0.1, value=1.0, label="Repetition Penalty"
    #         ),
    #     ],
    #     outputs=[
    #         gr.inputs.Textbox(
    #             lines=5,
    #             label="Output",
    #         )
    #     ],
    #     title="Chinese-Vicuna 中文小羊驼",
    #     description="中文小羊驼由各种高质量的开源instruction数据集，结合Alpaca-lora的代码训练而来，模型基于开源的llama7B，主要贡献是对应的lora模型。由于代码训练资源要求较小，希望为llama中文lora社区做一份贡献。",
    # ).launch(share=True)

    input_ids = tokenizer.encode("The mean of life is ", return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=36,
            top_p=0.95,
            temperature=0.8
        )
    print("*"*80)
    print("🦙:", tokenizer.decode([el.item() for el in generated_ids[0]]))


if __name__ == '__main__':
    main()
