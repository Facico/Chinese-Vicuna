import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
import json
import gradio as gr
import argparse
import warnings
import os
from datetime import datetime
from utils import StreamPeftGenerationMixin,StreamLlamaForCausalLM, printf
import utils
import copy
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import prompt

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default='')
parser.add_argument("--use_typewriter", type=int, default=1)
parser.add_argument("--prompt_type", type=str, default='chat')
parser.add_argument("--share_link", type=int, default=0)
parser.add_argument("--show_beam", type=int, default=0)
parser.add_argument("--int8", type=int, default=1)
args = parser.parse_args()
args.fix_token = True
printf('>>> args:', args)
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

LOAD_8BIT = args.int8
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path

# fix the path for local checkpoint
lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
if args.lora_path != '' and os.path.exists(args.lora_path):
    if not os.path.exists(lora_bin_path):
        pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
        printf('>>> load lora from', pytorch_bin_path)
        if os.path.exists(pytorch_bin_path):
            os.rename(pytorch_bin_path, lora_bin_path)
            warnings.warn(
                "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
            )
        else:
            assert ('Checkpoint is not Found!')
    else:
        printf('>>> load lora from', lora_bin_path)
else:
    printf('>>> load lora from huggingface url', args.lora_path)

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
    print(f'>>> load raw models from {BASE_MODEL}')
    if args.lora_path == "":
        model = StreamLlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )    
    else:
        print(f'>>> load lora models from {LORA_WEIGHTS}')
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        model = StreamPeftGenerationMixin.from_pretrained(
                model, LORA_WEIGHTS, torch_dtype=torch.float16, load_in_8bit=LOAD_8BIT,  device_map={"": 0}
        )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = StreamPeftGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = StreamPeftGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )
# fix tokenizer bug
if args.fix_token and tokenizer.eos_token_id != 2:
    warnings.warn(
        "The tokenizer eos token may be wrong. please check you llama-checkpoint"
    )
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2
model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def save(
    inputs,
    history,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=2.0,
    max_memory=1024,
    do_sample=False,
    prompt_type='0',
    **kwargs, 
):
    history = [] if history is None else history
    data_point = {}
    if prompt_type == 'instruct':
        PROMPT = prompt.instruct_prompt(tokenizer,max_memory)
    elif prompt_type == 'chat':
        PROMPT = prompt.chat_prompt(tokenizer,max_memory)
    else:
        raise Exception('not support')
    data_point['history'] = history
    # 实际上是每一步都可以不一样，这里只保存最后一步
    data_point['generation_parameter'] = {
        "temperature":temperature,
        "top_p":top_p,
        "top_k":top_k,
        "num_beams":num_beams,
        "bos_token_id":tokenizer.bos_token_id,
        "eos_token_id":tokenizer.eos_token_id,
        "pad_token_id":tokenizer.pad_token_id,
        "max_new_tokens":max_new_tokens,
        "min_new_tokens":min_new_tokens, 
        "do_sample":do_sample,
        "repetition_penalty":repetition_penalty,
        "max_memory":max_memory,
    }
    data_point['info'] = args.__dict__
    print(data_point)
    if args.int8:
        file_name = f"{args.lora_path}/{args.prompt_type.replace(' ','_')}_int8.jsonl"
    else:
        file_name = f"{args.lora_path}/{args.prompt_type.replace(' ','_')}_fp16.jsonl"
    utils.to_jsonl([data_point], file_name)

def evaluate(
    inputs,
    history,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=2.0,
    max_memory=1024,
    do_sample=False,
    prompt_type='0',
    **kwargs,
):
    history = [] if history is None else history
    data_point = {}
    if prompt_type == 'instruct':
        PROMPT = prompt.instruct_prompt(tokenizer,max_memory)
    elif prompt_type == 'chat':
        PROMPT = prompt.chat_prompt(tokenizer,max_memory)
    else:
        raise Exception('not support')
    
    data_point['history'] = copy.deepcopy(history)
    data_point['input'] = inputs

    input_ids = PROMPT.preprocess_gen(data_point)
    
    printf('------------------------------')
    printf(tokenizer.decode(input_ids))
    input_ids = torch.tensor([input_ids]).to(device) # batch=1

    printf('------------------------------')
    printf('shape',input_ids.size())
    printf('------------------------------')
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        do_sample=do_sample,
        bad_words_ids=tokenizer(['\n\nUser:','\n\nAssistant:'], add_special_tokens=False).input_ids,

        **kwargs,
    )
    
    return_text = [(item['input'], item['output']) for item in history]
    out_memory =False
    outputs = None
    with torch.no_grad():
        # 流式输出 / 打字机效果
        # streamly output / typewriter style
        if args.use_typewriter:
            try:
                for generation_output in model.stream_generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                    repetition_penalty=float(repetition_penalty),
                ):
                    gen_token = generation_output[0][-1].item()
                    printf(gen_token, end='(')
                    printf(tokenizer.decode(gen_token), end=') ')
                    
                    outputs = tokenizer.batch_decode(generation_output)
                    if args.show_beam:
                        show_text = "\n--------------------------------------------\n".join(
                            [ PROMPT.postprocess(output)+" ▌" for output in outputs]
                        )
                    else:
                        show_text = PROMPT.postprocess(outputs[0])+" ▌"
                    yield return_text +[(inputs, show_text)], history
            except torch.cuda.OutOfMemoryError:
                print('CUDA out of memory')
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                out_memory=True
            # finally only one
            printf('[EOS]', end='\n')
            show_text = PROMPT.postprocess(outputs[0] if outputs is not None else '### Response:')
            return_len = len(show_text)
            if out_memory==True:
                out_memory=False
                show_text+= '<p style="color:#FF0000"> [GPU Out Of Memory] </p> '
            if return_len > 0:
                output = PROMPT.postprocess(outputs[0], render=False)
                history.append({
                    'input': inputs,
                    'output': output,
                })

            return_text += [(inputs, show_text)]
            yield return_text, history
        # common 
        else:
            try:
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
                output = PROMPT.postprocess(output)
                history.append({
                    'input': inputs,
                    'output': output,
                })
                return_text += [(inputs, output)]
                yield return_text, history
            except torch.cuda.OutOfMemoryError:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                show_text = '<p style="color:#FF0000"> [GPU Out Of Memory] </p> '
                printf(show_text)
                return_text += [(inputs, show_text)]
                yield return_text, history

def clear():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return None, None


# gr.Interface对chatbot的clear有bug，因此我们重新实现了一个基于gr.block的UI逻辑
# gr.Interface has bugs to clear chatbot's history,so we customly implement it based on gr.block
with gr.Blocks() as demo:
    fn = evaluate
    title = gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 1rem'>"
        + "Chinese-Vicuna 中文小羊驼"
        + "</h1>"
    )
    description = gr.Markdown(
        "中文小羊驼由各种高质量的开源instruction数据集，结合Alpaca-lora的代码训练而来，模型基于开源的llama7B，主要贡献是对应的lora模型。由于代码训练资源要求较小，希望为llama中文lora社区做一份贡献。"
    )
    history = gr.components.State()
    with gr.Row().style(equal_height=False):
        with gr.Column(variant="panel"):
            input_component_column = gr.Column()
            with input_component_column:
                input = gr.components.Textbox(
                    lines=2, label="Input", placeholder="请输入问题."
                )
                temperature = gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Temperature")
                topp = gr.components.Slider(minimum=0, maximum=1, value=0.9, label="Top p")
                topk = gr.components.Slider(minimum=0, maximum=100, step=1, value=60, label="Top k")
                beam_number = gr.components.Slider(minimum=1, maximum=10, step=1, value=4, label="Beams Number")
                max_new_token = gr.components.Slider(
                    minimum=1, maximum=2048, step=1, value=256, label="Max New Tokens"
                )
                min_new_token = gr.components.Slider(
                    minimum=1, maximum=1024, step=1, value=5, label="Min New Tokens"
                )
                repeat_penal = gr.components.Slider(
                    minimum=0.1, maximum=10.0, step=0.1, value=2.0, label="Repetition Penalty"
                )
                max_memory = gr.components.Slider(
                    minimum=0, maximum=2048, step=1, value=2048, label="Max Memory"
                )
                do_sample = gr.components.Checkbox(label="Use sample")
                # must be str, not number !
                type_of_prompt = gr.components.Dropdown(
                    ['instruct', 'chat'], value=args.prompt_type, label="Prompt Type", info="select the specific prompt; use after clear history"
                )
                input_components = [
                    input, history, temperature, topp, topk, beam_number, max_new_token, min_new_token, repeat_penal, max_memory, do_sample, type_of_prompt
                ]
                input_components_except_states = [input, temperature, topp, topk, beam_number, max_new_token, min_new_token, repeat_penal, max_memory, do_sample, type_of_prompt]
            with gr.Row():
                cancel_btn = gr.Button('Cancel')
                submit_btn = gr.Button("Submit", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop", visible=False)
            with gr.Row():
                reset_btn = gr.Button("Reset Parameter")
                clear_history = gr.Button("Clear History")


        with gr.Column(variant="panel"):
            chatbot = gr.Chatbot().style(height=1024)
            output_components = [ chatbot, history ]  
            with gr.Row():
                save_btn = gr.Button("Save Chat")
        def wrapper(*args):
            # here to support the change between the stop and submit button
            try:
                for output in fn(*args):
                    output = [o for o in output]
                    # output for output_components, the rest for [button, button]
                    yield output + [
                        gr.Button.update(visible=False),
                        gr.Button.update(visible=True),
                    ]
            finally:
                yield [{'__type__': 'generic_update'}, {'__type__': 'generic_update'}] + [ gr.Button.update(visible=True), gr.Button.update(visible=False)]

        def cancel(history, chatbot):
            if history == []:
                return (None, None)
            return history[:-1], chatbot[:-1]

        extra_output = [submit_btn, stop_btn]
        save_btn.click(
            save, 
            input_components, 
            None, 
        )
        pred = submit_btn.click(
            wrapper, 
            input_components, 
            output_components + extra_output, 
            api_name="predict",
            scroll_to_output=True,
            preprocess=True,
            postprocess=True,
            batch=False,
            max_batch_size=4,
        )
        submit_btn.click(
            lambda: (
                submit_btn.update(visible=False),
                stop_btn.update(visible=True),
            ),
            inputs=None,
            outputs=[submit_btn, stop_btn],
            queue=False,
        )
        stop_btn.click(
            lambda: (
                submit_btn.update(visible=True),
                stop_btn.update(visible=False),
            ),
            inputs=None,
            outputs=[submit_btn, stop_btn],
            cancels=[pred],
            queue=False,
        )
        cancel_btn.click(
            cancel,
            inputs=[history, chatbot],
            outputs=[history, chatbot]
        )
        reset_btn.click(
            None, 
            [],
            (
                # input_components ; don't work for history...
                input_components_except_states
                + [input_component_column]
            ),  # type: ignore
            _js=f"""() => {json.dumps([
                getattr(component, "cleared_value", None) for component in input_components_except_states ] 
                + ([gr.Column.update(visible=True)])
                + ([])
            )}
            """,
        )
        clear_history.click(clear, None, [history, chatbot], queue=False)

demo.queue().launch(share=args.share_link)