import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
import json
import gradio as gr
import argparse
import warnings
import os
from utils import SteamGenerationMixin, printf
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default="./lora-Vicuna/checkpoint-3000")
parser.add_argument("--use_typewriter", type=int, default=1)
parser.add_argument("--share_link", type=int, default=0)
parser.add_argument("--use_local", type=int, default=1)
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path

# fix the path for local checkpoint
lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path) and args.use_local:
    pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
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


def generate_prompt_and_tokenize0(data_point, maxlen):
    # cutoff the history to avoid exceeding length limit
    init_prompt = PROMPT_DICT['prompt']
    init_ids = tokenizer(init_prompt)['input_ids']
    seqlen = len(init_ids)
    input_prompt = PROMPT_DICT['input'].format_map(data_point)
    input_ids = tokenizer(input_prompt)['input_ids']
    seqlen += len(input_ids)
    if seqlen > maxlen:
        raise Exception('>>> The input question is too long! Cosidering increase the Max Memory value or decrease the length of input! ')
    history_prompt = ''
    for history in data_point['history']:
        history_prompt+= PROMPT_DICT['history'].format_map(history) 
    # cutoff
    history_ids = tokenizer(history_prompt)['input_ids'][-(maxlen - seqlen):]
    input_ids = init_ids + history_ids + input_ids
    return input_ids

def postprocess0(text, render=True):
    # clip user
    text = text.split("### Assistant:")[1].strip()
    text = text.replace('�','').replace("Belle", "Vicuna")
    return text

def generate_prompt_and_tokenize1(data_point, maxlen):
    input_prompt = "\n".join(["User:" + i['input']+"\n"+"Assistant:" + i['output'] for i in data_point['history']]) + "\nUser:" + data_point['input'] + "\nAssistant:"
    if len(input_prompt) > maxlen:
        input_prompt = input_prompt[-maxlen:]
    input_prompt = PROMPT_DICT['prompt'].format_map({'input':input_prompt})
    input_ids = tokenizer(input_prompt)["input_ids"]
    return input_ids

def postprocess1(text, render=True):
    output = text.split("### Response:")[1].strip()
    output = output.replace("Belle", "Vicuna")
    printf('>>> output:', output)
    if '###' in output:
        output = output.split("###")[0]
    if 'User' in output:
        output = output.split("User")[0]
    output = output.replace('�','') 
    if render:
        # fix gradio chatbot markdown code render bug
        lines = output.split("\n")
        for i, line in enumerate(lines):
            if "```" in line:
                if line != "```":
                    lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                else:
                    lines[i] = '</code></pre>'
            else:
                if i > 0:
                    lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
        output =  "".join(lines)
        # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
    return output

PROMPT_DICT0 = {
    'prompt': (
        "The following is a conversation between an AI assistant called Assistant and a human user called User."
        "Assistant is is intelligent, knowledgeable, wise and polite.\n\n"
    ),
    'history': (
        "User:{input}\n\nAssistant:{output}\n\n"
    ),
    'input': (
        "User:{input}\n\n### Assistant:"
    ),
    'preprocess': generate_prompt_and_tokenize0,
    'postprocess': postprocess0,
}
PROMPT_DICT1 = {
    'prompt': (
        "The following is a conversation between an AI assistant called Assistant and a human user called User.\n\n"
        "### Instruction:\n{input}\n\n### Response:"
    ),
    'preprocess': generate_prompt_and_tokenize1,
    'postprocess': postprocess1,
}
PROMPT_DICT = None

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)



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
    global PROMPT_DICT
    if prompt_type == '0':
        PROMPT_DICT = PROMPT_DICT0
    elif prompt_type == '1':
        PROMPT_DICT = PROMPT_DICT1
    else:
        raise Exception('not support')
    
    history = [] if history is None else history
    data_point = {
        'history': history,
        'input': inputs,
    }
    printf(data_point)
    input_ids = PROMPT_DICT['preprocess'](data_point, max_memory)
    printf('>>> input prompts:', tokenizer.decode(input_ids))
    input_ids = torch.tensor([input_ids]).to(device) # batch=1
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
        do_sample=do_sample,
        **kwargs,
    )
    
    return_text = [(item['input'], item['output']) for item in history]
    
    with torch.no_grad():
        # 流式输出 / 打字机效果
        # streamly output / typewriter style
        if args.use_typewriter:
            for generation_output in model.stream_generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                repetition_penalty=float(repetition_penalty),
            ):
                outputs = tokenizer.batch_decode(generation_output)
                show_text = "\n--------------------------------------------\n".join(
                    [PROMPT_DICT['postprocess'](output)+" ▌" for output in outputs]
                )
                printf(show_text)
                yield return_text +[(inputs, show_text)], history
            # finally only one
            show_text = PROMPT_DICT['postprocess'](outputs[0])
            printf(show_text)
            output = PROMPT_DICT['postprocess'](outputs[0], render=False)
            history.append({
                'input': inputs,
                'output': output,
            })
            return_text += [(inputs, show_text)]
            yield return_text, history
        # common 
        else:
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
            output = PROMPT_DICT['postprocess'](output)
            history.append({
                'input': inputs,
                'output': output,
            })
            return_text += [(inputs, output)]
            yield return_text, history

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
                    minimum=1, maximum=2000, step=1, value=256, label="Max New Tokens"
                )
                min_new_token = gr.components.Slider(
                    minimum=1, maximum=100, step=1, value=5, label="Min New Tokens"
                )
                repeat_penal = gr.components.Slider(
                    minimum=0.1, maximum=10.0, step=0.1, value=2.0, label="Repetition Penalty"
                )
                max_memory = gr.components.Slider(
                    minimum=0, maximum=2048, step=1, value=256, label="Max Memory"
                )
                do_sample = gr.components.Checkbox(label="Use sample")
                # must be str, not number !
                type_of_prompt = gr.components.Dropdown(
                    ['0', '1'], value='1', label="Prompt Type", info="select the specific prompt; use after clear history"
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
        clear_history.click(lambda: (None, None), None, [history, chatbot], queue=False)

demo.queue().launch(share=args.share_link!=0, inbrowser=True)