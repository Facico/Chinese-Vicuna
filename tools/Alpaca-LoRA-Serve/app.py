from strings import TITLE, ABSTRACT, BOTTOM_LINE
from strings import DEFAULT_EXAMPLES
from strings import SPECIAL_STRS
from styles import PARENT_BLOCK_CSS

import time
import gradio as gr

from args import parse_args
from model import load_model
from gen import get_output_batch, StreamModel
from utils import generate_prompt, post_processes_batch, post_process_stream, get_generation_config, common_post_process
from transformers import GenerationConfig

def chat_stream(
    context,
    instruction,
    state_chatbot,
    max_tokens=256,
    top_p=0.9, 
    temperature=1.0, 
    top_k=40, 
    num_beams=2,
    repetition_penalty=3.0,
):
    # print(instruction)

    # user input should be appropriately formatted (don't be confused by the function name)
    instruction_display = common_post_process(instruction)
    instruction_prompt = generate_prompt(instruction, state_chatbot, context)  
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        use_cache=True,
        min_length=0,
        max_length=1000,
    )  
    model.generation_config = generation_config
    bot_response = model(
        instruction_prompt,
        max_tokens=max_tokens,
        #repetition_penalty=float(repetition_penalty),
    )
    
    instruction_display = None if instruction_display == SPECIAL_STRS["continue"] else instruction_display
    state_chatbot = state_chatbot + [(instruction_display, None)]
    
    prev_index = 0
    agg_tokens = ""
    cutoff_idx = 0
    for tokens in bot_response:
        tokens = tokens.strip()
        cur_token = tokens[prev_index:]
        
        if "#" in cur_token and agg_tokens == "":
            cutoff_idx = tokens.find("#")
            agg_tokens = tokens[cutoff_idx:]

        if agg_tokens != "":
            if len(agg_tokens) < len("### Instruction:") :
                agg_tokens = agg_tokens + cur_token
            elif len(agg_tokens) >= len("### Instruction:"):
                if tokens.find("### Instruction:") > -1:
                    processed_response, _ = post_process_stream(tokens[:tokens.find("### Instruction:")].strip())

                    state_chatbot[-1] = (
                        instruction_display, 
                        processed_response
                    )
                    yield (state_chatbot, state_chatbot, context)
                    break
                else:
                    agg_tokens = ""
                    cutoff_idx = 0

        if agg_tokens == "":
            processed_response, to_exit = post_process_stream(tokens)
            state_chatbot[-1] = (instruction_display, processed_response)
            yield (state_chatbot, state_chatbot, context)

            if to_exit:
                break

        prev_index = len(tokens)

    yield (
        state_chatbot,
        state_chatbot,
        gr.Textbox.update(value=tokens) if instruction_display == SPECIAL_STRS["summarize"] else context
    )

def chat_batch(
    contexts,
    instructions, 
    state_chatbots,
    max_tokens=256,
    top_p=0.9, 
    temperature=1.0, 
    top_k=40, 
    num_beams=2,
    repetition_penalty=3.0,
):
    state_results = []
    ctx_results = []

    instruct_prompts = [
        generate_prompt(instruct, histories, ctx) 
        for ctx, instruct, histories in zip(contexts, instructions, state_chatbots)
    ]
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        use_cache=True,
        min_length=0,
        max_length=1000,
    )   
    bot_responses = get_output_batch(
        model, tokenizer, instruct_prompts, generation_config, max_tokens, repetition_p
    )
    bot_responses = post_processes_batch(bot_responses)

    for ctx, instruction, bot_response, state_chatbot in zip(contexts, instructions, bot_responses, state_chatbots):
        new_state_chatbot = state_chatbot + [('' if instruction == SPECIAL_STRS["continue"] else instruction, bot_response)]
        ctx_results.append(gr.Textbox.update(value=bot_response) if instruction == SPECIAL_STRS["summarize"] else ctx)
        state_results.append(new_state_chatbot)

    return (state_results, state_results, ctx_results)

def reset_textbox():
    return gr.Textbox.update(value='')

def run(args):
    global model, tokenizer, generation_config, batch_enabled
    
    batch_enabled = True if args.batch_size > 1 else False    

    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url
    )    
    
    if not batch_enabled:
        model = StreamModel(model, tokenizer)
        # model.generation_config = generation_config
    
    with gr.Blocks(css=PARENT_BLOCK_CSS) as demo:
        state_chatbot = gr.State([])

        with gr.Column(elem_id='col_container'):
            gr.Markdown(f"## {TITLE}\n\n\n{ABSTRACT}")

            with gr.Column(scale=1):
                max_length = gr.Slider(0, 2048, value=128, step=1, label="max_tokens", interactive=True)
                top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
                top_k = gr.Slider(0, 100, value=40, step=1, label="Top k", interactive=True)
                num_beams = gr.Slider(minimum=1, maximum=8, step=1, value=2, label="Beams", interactive=True)
                repetition_penalty = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=3.0, label="Repetition Penalty")
                
            with gr.Accordion("Context Setting", open=False):
                context_txtbox = gr.Textbox(placeholder="Surrounding information to AI", label="Enter Context")
                hidden_txtbox = gr.Textbox(placeholder="", label="Order", visible=False)

            chatbot = gr.Chatbot(elem_id='chatbot', label="Alpaca-LoRA")
            instruction_txtbox = gr.Textbox(placeholder="What do you want to say to AI?", label="Instruction")
            send_prompt_btn = gr.Button(value="Send Prompt")
            
            with gr.Accordion("Helper Buttons", open=False):
                gr.Markdown(f"`Continue` lets AI to complete the previous incomplete answers. `Summarize` lets AI to summarize the conversations so far.")
                continue_txtbox = gr.Textbox(value=SPECIAL_STRS["continue"], visible=False)
                summrize_txtbox = gr.Textbox(value=SPECIAL_STRS["summarize"], visible=False)
                
                continue_btn = gr.Button(value="Continue")
                summarize_btn = gr.Button(value="Summarize")

            gr.Markdown("#### Examples")
            for idx, examples in enumerate(DEFAULT_EXAMPLES):
                with gr.Accordion(examples["title"], open=False):
                    gr.Examples(
                        examples=examples["examples"], 
                        inputs=[
                            hidden_txtbox, instruction_txtbox
                        ],
                        label=None
                    )

            gr.Markdown(f"{BOTTOM_LINE}")
            
        send_prompt_btn.click(
            chat_batch if batch_enabled else chat_stream, 
            [context_txtbox, instruction_txtbox, state_chatbot, max_length, top_p, temperature, top_k, num_beams, repetition_penalty],
            [state_chatbot, chatbot, context_txtbox],
            batch=batch_enabled,
            max_batch_size=args.batch_size,
        )
        send_prompt_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )
        
        continue_btn.click(
            chat_batch if batch_enabled else chat_stream, 
            [context_txtbox, continue_txtbox, state_chatbot, max_length, top_p, temperature, top_k, num_beams, repetition_penalty],
            [state_chatbot, chatbot, context_txtbox],
            batch=batch_enabled,
            max_batch_size=args.batch_size,
        )
        continue_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )
        
        summarize_btn.click(
            chat_batch if batch_enabled else chat_stream, 
            [context_txtbox, summrize_txtbox, state_chatbot, max_length, top_p, temperature, top_k, num_beams, repetition_penalty],
            [state_chatbot, chatbot, context_txtbox],
            batch=batch_enabled,
            max_batch_size=args.batch_size,
        )
        summarize_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )              

    demo.queue(
        concurrency_count=2,
        max_size=100,
        api_open=False if args.api_open == "no" else True
    ).launch(
        max_threads=2,
        share=False if args.share == "no" else True,
        server_port=args.port,
        server_name="0.0.0.0",
    )

if __name__ == "__main__":
    args = parse_args()
    run(args)
