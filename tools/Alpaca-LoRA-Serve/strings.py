TITLE = "Alpaca-LoRA Playground"

ABSTRACT = """
Thanks to [tolen](https://github.com/tloen/alpaca-lora), this simple application runs Alpaca-LoRA which is instruction fine-tuned version of [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) from Meta AI. Alpaca-LoRA is *Low-Rank LLaMA Instruct-Tuning* which is inspired by [Stanford Alpaca project](https://github.com/tatsu-lab/stanford_alpaca). This demo application currently runs 30B version on A6000.

We are thankful to the [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU instances. 
"""

BOTTOM_LINE = """

This demo application runs the open source project, [Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve). By default, it runs with streaming mode, but you can also run with dynamic batch generation model. Please visit the repo, find more information, and contribute if you can.

Alpaca-LoRA is built on the same concept as Standford Alpaca project, but it lets us train and inference on a smaller GPUs such as RTX4090 for 7B version. Also, we could build very small size of checkpoints on top of base models thanks to [🤗 transformers](https://huggingface.co/docs/transformers/index), [🤗 peft](https://github.com/huggingface/peft), and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes/tree/main) libraries.
"""

DEFAULT_EXAMPLES = [
    {
        "title": "1️⃣ List all Canadian provinces in alphabetical order.",
        "examples": [
            ["1", "List all Canadian provinces in alphabetical order."],
            ["2", "Which ones are on the east side?"],
            ["3", "What foods are famous in each province on the east side?"],
            ["4", "What about sightseeing? or landmarks? list one per province"],
        ],
    },
    {
        "title": "2️⃣ Tell me about Alpacas.",
        "examples": [
            ["1", "Tell me about alpacas in two sentences"],
            ["2", "What other animals are living in the same area?"],
            ["3", "Are they the same species?"],
            ["4", "Write a Python program to return those species"],
        ],
    },
    {
        "title": "3️⃣ Tell me about the king of France in 2019.",
        "examples": [
            ["1", "Tell me about the king of France in 2019."],
            ["2", "What about before him?"],
        ]
    },
    {
        "title": "4️⃣ Write a Python program that prints the first 10 Fibonacci numbers.",
        "examples": [
            ["1", "Write a Python program that prints the first 10 Fibonacci numbers."],
            ["2", "Could you explain how the code works?"],
            ["3", "What is recursion?"],
        ]
    }
]

SPECIAL_STRS = {
    "continue": "continue.",
    "summarize": "summarize our conversations so far in three sentences."
}