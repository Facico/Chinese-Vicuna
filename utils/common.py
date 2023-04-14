import os
from typing import Optional, Union, overload

def printf(*args):
    if os.environ.get('DEBUG',False):
        print('>>> ', *args)

# region generating prompt

def common_generate_prompt(
        instruct_or_dict: Union[str, dict], input: Optional[str] = None,*,
        with_input: str, without_input:str):
    if isinstance(instruct_or_dict, dict):
        input = instruct_or_dict["input"]
        instruction = instruct_or_dict["instruction"]
    else:
        instruction = instruct_or_dict
    if input:
        return with_input.format(instruction, input)
    else:
        return without_input.format(instruction)

TASK_PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:"""

TASK_PROMPT_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:"""

@overload
def task_generate_prompt(instruct_or_dict: dict) -> str: ...

@overload
def task_generate_prompt(instruction:str, input:str) -> str: ...

def task_generate_prompt(instruct_or_dict: Union[str, dict], input: Optional[str] = None):
    return common_generate_prompt(
        instruct_or_dict, input,
        with_input=TASK_PROMPT_WITH_INPUT, 
        without_input=TASK_PROMPT_WITHOUT_INPUT
    )


AI_PROMPT_WITH_INPUT = """The following is a conversation between an AI assistant called Assistant and a human user called User.

### Instruction:
{}

### Input:
{}

### Response:"""

AI_PROMPT_WITHOUT_INPUT = """The following is a conversation between an AI assistant called Assistant and a human user called User.

### Instruction:
{}

### Response:
"""

def ai_generate_prompt(instruct_or_dict: Union[str, dict], input: Optional[str] = None):
    return common_generate_prompt(
        instruct_or_dict, input, 
        with_input=AI_PROMPT_WITH_INPUT, 
        without_input=AI_PROMPT_WITHOUT_INPUT
    )
# endregion generating prompt