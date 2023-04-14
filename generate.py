import sys
import torch
import transformers
import gradio as gr
import warnings
import os

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    HfArgumentParser,
)

from utils import (
    SteamGenerationMixin,
    ModelArguments,
    ChatArguments,
    task_generate_prompt as generate_prompt,
)

hf_parser = HfArgumentParser((ModelArguments, ChatArguments))
model_args: ModelArguments
chat_args: ChatArguments
model_args, chat_args = hf_parser.parse_args_into_dataclasses()

tokenizer = LlamaTokenizer.from_pretrained(model_args.token_path if model_args.token_path else model_args.model_path)

LOAD_8BIT = model_args.load_8bit
BASE_MODEL = model_args.model_path
LORA_WEIGHTS = model_args.lora_path


# fix the path for local checkpoint
lora_bin_path = os.path.join(LORA_WEIGHTS, "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path) and model_args.use_local:
    pytorch_bin_path = os.path.join(LORA_WEIGHTS, "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
        os.rename(pytorch_bin_path, lora_bin_path)
        warnings.warn(
            "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
        )
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

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model = SteamGenerationMixin.from_pretrained(
        model, LORA_WEIGHTS, torch_dtype=torch.float16, device_map={"": 0}
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = SteamGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = SteamGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=2.0,
    **kwargs,
):
    prompt = generate_prompt(input, None)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        **kwargs,
    )
    with torch.no_grad():
        if chat_args.use_typewriter:
            for generation_output in model.stream_generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                repetition_penalty=float(repetition_penalty),
            ):
                outputs = tokenizer.batch_decode(generation_output)
                show_text = "\n--------------------------------------------\n".join(
                    [output.split("### Response:")[1].strip().replace('�','')+" ▌" for output in outputs]
                )
                # if show_text== '':
                #     yield last_show_text
                # else:
                yield show_text
            yield outputs[0].split("### Response:")[1].strip().replace('�','')
        else:
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                repetition_penalty=1.3,
            )
            output = generation_output.sequences[0]
            output = tokenizer.decode(output).split("### Response:")[1].strip()
            print(output)
            yield output


gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Input", placeholder="Tell me about alpacas."
        ),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=10, step=1, value=4, label="Beams Number"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=256, label="Max New Tokens"
        ),
        gr.components.Slider(
            minimum=1, maximum=100, step=1, value=1, label="Min New Tokens"
        ),
        gr.components.Slider(
            minimum=0.1, maximum=10.0, step=0.1, value=2.0, label="Repetition Penalty"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=25,
            label="Output",
        )
    ],
    title="Chinese-Vicuna 中文小羊驼",
    description="中文小羊驼由各种高质量的开源instruction数据集，结合Alpaca-lora的代码训练而来，模型基于开源的llama7B，主要贡献是对应的lora模型。由于代码训练资源要求较小，希望为llama中文lora社区做一份贡献。",
).queue().launch(share=True)
