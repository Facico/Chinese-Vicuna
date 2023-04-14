from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

# region dataclasses for arguments parsing


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="decapoda-research/llama-7b-hf")
    token_path: Optional[str] = field(default=None, metadata={
        "help": "load tokenizer from model_path is it is None else use this path to load tokenizer"
    })
    lora_path: Optional[str] = field(default="./lora-Vicuna/checkpoint-3000")
    use_local: Optional[int] = field(default=1)
    load_8bit: Optional[bool] = field(default=True)
    max_length: int = field(default=256)


@dataclass
class ChatArguments:
    use_typewriter: int = 1
    share_link: int = 0


@dataclass
class FinetuningArguments(TrainingArguments):
    # customized fields
    data_path: Optional[str] = field(default="merge.json")
    test_size: int = 200
    wandb: bool = field(default=False, metadata={
                        "help": "wandb=True means report_to=wandb"})

    # set default fields
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=128//4)
    num_train_epochs: float = field(default=3.0)
    max_steps: int = field(default=-1)
    learning_rate: float = field(default=3e-4)
    warmup_steps = 100
    fp16: bool = field(default=True)
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "load lora_model if not None"})
    logging_steps: bool = field(default=20)
    evaluation_strategy: str = "steps" if test_size > 0 else "no"
    save_strategy: str = field(default="steps")
    eval_steps: int = field(default=200) if test_size > 0 else None
    save_steps = 200
    save_total_limit = 30
    load_best_model_at_end = True if test_size > 0 else False
    output_dir: str = "lora-Vicuna"
    report_to = "wandb" if wandb else None
    ignore_data_skip: bool = field(default=False)


@dataclass
class EvalArguments:
    lora_path: Optional[str] = field(default="./lora-Vicuna/checkpoint-3000")
    use_local: Optional[int] = field(default=1)
    wandb: Optional[bool] = field(default=False)


@dataclass
class MiscArguments:
    """self-customized arguments here
    """
# endregion dataclasses for arguments parsing
