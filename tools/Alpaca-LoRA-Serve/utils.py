import re
import yaml

from transformers import GenerationConfig

from strings import SPECIAL_STRS
from constants import num_of_characters_to_keep
from constants import html_tag_pattern, multi_line_pattern, multi_space_pattern
from constants import repl_empty_str, repl_br_tag, repl_span_tag_multispace, repl_linebreak

def get_generation_config(path):
    with open(path, 'rb') as f:
        generation_config = yaml.safe_load(f.read())

    return GenerationConfig(**generation_config["generation_config"])

def generate_prompt(prompt, histories, ctx=None):
    convs = f"""Below is a history of instructions that describe tasks, paired with an input that provides further context. Write a response that appropriately completes the request by remembering the conversation history.
    
"""
    if ctx is not None:
        convs = f"""{ctx}

"""
    
    start_idx = 0
    
    for idx, history in enumerate(histories):
        history_prompt = history[0]
        if history_prompt == SPECIAL_STRS["summarize"]:
            start_idx = idx

    # drop the previous conversations if user has summarized
    for history in histories[start_idx if start_idx == 0 else start_idx+1:]:
        history_prompt = history[0]
        history_response = history[1]
        
        history_response = history_response.replace("<br>", "\n")
        history_response = re.sub(
            html_tag_pattern, repl_empty_str, history_response
        )

        convs = convs + f"""### Instruction:{history_prompt}

### Response:{history_response}

"""

    convs = convs + f"""### Instruction:{prompt}

### Response:"""

    return convs[-num_of_characters_to_keep:]

# applicable to instruction to be displayed as well
def common_post_process(original_str):
    original_str = re.sub(
        multi_line_pattern, repl_br_tag, original_str
    )
    original_str = re.sub(
        multi_space_pattern, repl_span_tag_multispace, original_str
    )
    
    return original_str

def post_process_stream(bot_response):
    # sometimes model spits out text containing 
    # "### Response:" and "### Instruction: -> in this case, we want to stop generating
    if "### Response:" in bot_response or "### Input:" in bot_response:
        bot_response = bot_response.replace("### Response:", '').replace("### Input:", '').strip()
        return bot_response, True
    
    return common_post_process(bot_response), False

def post_process_batch(bot_response):
    bot_response = bot_response.split("### Response:")[-1].strip()
    return common_post_process(bot_response)

def post_processes_batch(bot_responses):
    return [post_process_batch(r) for r in bot_responses]
