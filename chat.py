import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from accelerate import dispatch_model, infer_auto_device_map
from peft.utils import PeftType, set_peft_model_state_dict
import copy
import transformers
import json
import gradio as gr
import argparse
import warnings
import os
import torch.distributed as dist
from typing import Optional, Tuple, Union, List, Callable
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.utils import (
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
    GenerationMixin,
)
from torch import nn

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


class SteamGenerationMixin(PeftModelForCausalLM, GenerationMixin):
    # support for streamly beam search
    @torch.no_grad()
    def stream_generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        **kwargs,
    ):
        self._reorder_cache = self.base_model._reorder_cache
        if is_deepspeed_zero3_enabled() and dist.world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

        if kwargs.get("attention_mask", None) is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(
                kwargs["input_ids"].shape[0], self.peft_config.num_virtual_tokens
            ).to(kwargs["input_ids"].device)
            kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, kwargs["attention_mask"]), dim=1
            )
        if kwargs.get("position_ids", None) is not None:
            warnings.warn(
                "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
            )
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
            )
            kwargs["token_type_ids"] = None

        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)

        bos_token_id, eos_token_id, pad_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
            generation_config.pad_token_id,
        )

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = (
                generation_config.max_new_tokens + input_ids_seq_length
            )
        if generation_config.min_new_tokens is not None:
            generation_config.min_length = (
                generation_config.min_new_tokens + input_ids_seq_length
            )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = (
                "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            )

        # 2. Set generation parameters if not already defined
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )
        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        # 10. go into beam search generation modes
        # 11. prepare beam search scorer
        num_beams = generation_config.num_beams
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=input_ids.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # beam_search logits
        batch_beam_size, cur_len = input_ids.shape
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len) hack: adjust tokens for Marian.
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[
                :, None
            ].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=None,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            # increase cur_len
            cur_len = cur_len + 1

            yield input_ids

            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        final_result = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=None,
        )
        yield final_result["sequences"]

    # default it call `model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config)`, not cls!! so inherent PeftModelForCausalLM is no sense
    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        # load the config
        config = LoraConfig.from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        # here is the hack
        model = cls(model, config)

        # load weights if any
        if os.path.exists(os.path.join(model_id, "adapter_model.bin")):
            filename = os.path.join(model_id, "adapter_model.bin")
        else:
            try:
                filename = hf_hub_download(model_id, "adapter_model.bin")
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {'adapter_model.bin'} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)
        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)
            if model.peft_config.peft_type == PeftType.LORA:
                add_hook_to_module(model.base_model.model, hook)
            else:
                remove_hook_from_submodules(model.prompt_encoder)
                add_hook_to_module(model.base_model, hook)
        return model


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default="./lora-Vicuna/checkpoint-3000")
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


CHAT_DICT = {
    'prompt': (
        "The following is a conversation between an AI assistant called Assistant and a human user called User."
        "Bot is is intelligent, knowledgeable, wise and polite.\n\n"
    ),
    'history': (
        "User:\n{input}\n\Assistant:{output}\n\n"
    ),
    'input': (
        "### User:\n{input}\n\n### Assistant:"
    )
}

def generate_prompt_and_tokenize(data_point, maxlen):
    # cutoff the history to avoid exceeding length limit
    init_prompt = CHAT_DICT['prompt']
    init_ids = tokenizer(init_prompt)['input_ids']
    seqlen = len(init_ids)
    input_prompt = CHAT_DICT['input'].format_map(data_point)
    input_ids = tokenizer(input_prompt)['input_ids']
    seqlen += len(input_ids)
    if seqlen > maxlen:
        raise Exception('>>> The input question is too long! Cosidering increase the Max Memory value or decrease the length of input! ')
    history_prompt = ''
    for history in data_point['history']:
        history_prompt+= CHAT_DICT['history'].format_map(history) 
    # cutoff
    history_ids = tokenizer(history_prompt)['input_ids'][-(maxlen - seqlen):]
    input_ids = init_ids + history_ids + input_ids
    return input_ids

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    input,
    history,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=2.0,
    max_memory=1024,
    **kwargs,
):
    
    history = [] if history is None else history
    data_point = {
        'history': history,
        'input': input,
    }
    print(data_point)
    input_ids = generate_prompt_and_tokenize(data_point, max_memory)
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
        **kwargs,
    )
    
    return_text = [(item['input'], item['output']) for item in history]
    
    with torch.no_grad():
        for generation_output in model.stream_generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=float(repetition_penalty),
        ):
            def clip_redundant_user(text):
                if "User:" in text:
                    text = text.split("User:")[0]
                return text
            outputs = tokenizer.batch_decode(generation_output)
            show_text = "\n--------------------------------------------\n".join(
                [clip_redundant_user(output.split("### Assistant:")[1].strip().replace("\n\n###", "").replace('�','').replace("Belle", "Vicuna")) for output in outputs]
            )
            yield return_text +[(input, show_text)], history
        
        history.append({
            'input': input,
            'output': show_text,
        })
        return_text += [(input, show_text)]
        yield return_text, history


# inputs = [
#     gr.components.Textbox(
#         lines=2, label="Input", placeholder="Tell me about alpacas."
#     ),
#     "state",
#     gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
#     gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
#     gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
#     gr.components.Slider(minimum=1, maximum=10, step=1, value=4, label="Beams Number"),
#     gr.components.Slider(
#         minimum=1, maximum=2000, step=1, value=256, label="Max New Tokens"
#     ),
#     gr.components.Slider(
#         minimum=1, maximum=100, step=1, value=5, label="Min New Tokens"
#     ),
#     gr.components.Slider(
#         minimum=0.1, maximum=10.0, step=0.1, value=2.0, label="Repetition Penalty"
#     ),
#     gr.components.Slider(
#         minimum=0, maximum=2048, step=1, value=256, label="Max Memory"
#     ),
# ]
# outputs = [
#     gr.Chatbot().style(height=750),
#     "state"
# ]
# evaluate_block = gr.Interface(
#     fn=evaluate,
#     inputs=inputs,
#     outputs=outputs,
#     allow_flagging="auto",
#     title="Chinese-Vicuna 中文小羊驼",
#     description="中文小羊驼由各种高质量的开源instruction数据集，结合Alpaca-lora的代码训练而来，模型基于开源的llama7B，主要贡献是对应的lora模型。由于代码训练资源要求较小，希望为llama中文lora社区做一份贡献。",
# ).queue().launch(share=False)

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
                beam_number = gr.components.Slider(minimum=2, maximum=10, step=1, value=4, label="Beams Number")
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
                input_components = [
                    input, history, temperature, topp, topk, beam_number, max_new_token, min_new_token, repeat_penal, max_memory
                ]
                input_components_except_states = [input, temperature, topp, topk, beam_number, max_new_token, min_new_token, repeat_penal, max_memory]
            with gr.Row():
                reset_btn = gr.Button("Reset")
                submit_btn = gr.Button("Submit", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop", visible=False)
            clear_history = gr.Button("Clear History")
            

        with gr.Column(variant="panel"):
            chatbot = gr.Chatbot().style(height=750)
            output_components = [ chatbot, history ]  
            # clear_chatbot = gr.Button("Clear ChatBot")
        # Wrap the original function to show/hide the "Stop" button
        def wrapper(*args):
            # The main idea here is to call the original function
            # and append some updates to keep the "Submit" button
            # hidden and the "Stop" button visible
            # The 'finally' block hides the "Stop" button and
            # shows the "submit" button. Having a 'finally' block
            # will make sure the UI is "reset" even if there is an exception
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
        # clear_history.click(
        #     lambda: [None,None], # or None, don't work 
        #     None, 
        #     [chatbot, history], 
        #     queue=False
        # )
        clear_history.click(lambda: (None, None), None, [history, chatbot], queue=False)
        # clear_chatbot.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch(share=True)