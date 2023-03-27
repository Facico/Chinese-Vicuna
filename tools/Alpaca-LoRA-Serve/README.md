# 🦙 🚀 Alpaca-LoRA as a Chatbot Service

🚧 This project is still under development process. While serving the project, I noticed there are some bugs emitted by the model itself such as too many line breaks which causes OOM eventually. You can propose PR, but I will merge any improvement at any time as soon as I spot any problems.

🔗 **Demo link**: [Batch Mode](https://notebooksf.jarvislabs.ai/43j3x9FSS8Tg0sqvMlDgKPo9vsoSTTKRsX4RIdC3tNd6qeQ6ktlA0tyWRAR3fe_l) and [Streaming Mode](https://notebookse.jarvislabs.ai/BuOu_VbEuUHb09VEVHhfnFq4-PMhBRVCcfHBRCOrq7c4O9GI4dIGoidvNf76UsRL/) (both are running on a single A6000 instance)

The **easiest way** to run this project is to use Colab. Just open up the [alpaca_lora_in_colab](https://github.com/deep-diver/Alpaca-LoRA-Serve/blob/main/notebooks/alpaca_lora_in_colab.ipynb) notebook in Colab (there is a button `open in colab`), and run every cell sequentially. With the standard GPU instance(___T4___), you can run 7B and 13B models. With the premium GPU instance(___A100 40GB___), you can even run 30B model! Screenshot👇🏼 Just note that the connection could be somewhat unstable, so I recommend you to use Colab for development purpose.

![](https://i.ibb.co/hZ3771L/Screen-Shot-2023-03-22-at-9-36-15-PM.png)

This repository demonstrates Alpaca-LoRA as a Chatbot service with [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) and [Gradio](https://gradio.app/). It comes with the following features:

### Mode

**1. Batch Generation Mode**: batch generation mode aggregates requests up to `batch_size`, and pass the prompts in the requests to the model. It waits the current requests are fully handled. For instance, with `batch_size=4`, if a user sends a request, that is under processing. While it is under processing, if other users are connected, up to 4 requests from the users are aggregated and processed as soon as the current one is done.

**2. Streaming Mode**: streaming mode handles multiple requests in a interleaving way with threads. For instance, if there are two users (A and B) are connected, A's request is handled, and then B's request is handled, and then A's request is handled again.... This is because of the nature of streaming mode which generates and `yield` tokens in one by one manner. 

### Context management

- Alpaca-LoRA as a Chatbot Service manages context in two ways. First of all, it remembers(stores) every history of the conversations by default as in the following code snippet. `context_string` is set as ___"Below is a history of instructions that describe tasks, paired with an input that provides further context. Write a response that appropriately completes the request by remembering the conversation history."___ by default, but it could be set manually via the `Context` field on top of the screen. 
  - additionally, there is a `Summarize` button in the middle (you need to expand the component labeled as ___"Helper Buttons"___). If you click this button, it automatically input ___"summarize our conversations so far in three sentences."___ as a prompt, and the resulting generated text will be inserted into the `Context` field. THen all the conversation history up to this point will be ignored. That means the conversation fresh restarts with the below code snippet except `context_string` will be filled up with the model generated text.
  - _NOTE: the only last 2,000 characters are kept, and this number can be configured in `constants.py`_

```python
f"""{context_string}

### Input: {input} # Surrounding information to AI

### Instruction: {prompt1} # First instruction/prompt given by user

### Response {response1} # First response on the first prompt by AI

### Instruction: {prompt2} # Second instruction/prompt given by user

### Response: {response2} # Second response on the first prompt by AI
....
"""
```

### misc.

- There is a `continue` button in the middle of screen. What it does is to simply send ___"continue."___ prompt to the model. This is useful if you get incomplete previous response from the model. With the ___"continue."___, the model tries to complete the response. Also, since this is a continuation of the response, the ___"continue."___ prompt will be hidden to make chatting history more natural.

### Currently supported LoRA checkpoints
  - [tloen/alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b): the original 7B Alpaca-LoRA checkpoint by tloen
  - [chansung/alpaca-lora-13b](https://huggingface.co/chansung/alpaca-lora-13b): the 13B Alpaca-LoRA checkpoint by myself(chansung) with the same script to tune the original 7B model
  - [chansung/koalpaca-lora-13b](https://huggingface.co/chansung/koalpaca-lora-13b): the 13B Alpaca-LoRA checkpoint by myself(chansung) with the Korean dataset created by [KoAlpaca project](https://github.com/Beomi/KoAlpaca) by Beomi. It works for English(user) to Korean(AI) conversations.
  - [chansung/alpaca-lora-30b](https://huggingface.co/chansung/alpaca-lora-30b): the 30B Alpaca-LoRA checkpoint by myself(chansung) with the same script to tune the original 7B model

## Instructions

0. Prerequisites

Note that the code only works `Python >= 3.9`

```console
$ conda create -n alpaca-serve python=3.9
$ conda activate alpaca-serve
```

1. Install dependencies
```console
$ cd Alpaca-LoRA-Serve
$ pip install -r requirements.txt
```

2. Run Gradio application
```console
$ BASE_URL=decapoda-research/llama-7b-hf
$ FINETUNED_CKPT_URL=tloen/alpaca-lora-7b

$ python app.py --base_url $BASE_URL --ft_ckpt_url $FINETUNED_CKPT_URL --port 6006
```

the following flags are supported

```console
usage: app.py [-h] [--base_url BASE_URL] [--ft_ckpt_url FT_CKPT_URL] [--port PORT] [--batch_size BATCH_SIZE]
              [--api_open API_OPEN] [--share SHARE] [--gen_config_path GEN_CONFIG_PATH]

Gradio Application for Alpaca-LoRA as a chatbot service

optional arguments:
  -h, --help            show this help message and exit
  --base_url BASE_URL   Hugging Face Hub URL
  --ft_ckpt_url FT_CKPT_URL
                        Hugging Face Hub URL
  --port PORT           port number where the app is served
  --batch_size BATCH_SIZE
                        how many requests to handle at the same time
                        default is set to 1 which enables streaming mode
  --api_open API_OPEN   do you want to open as API
  --share SHARE         do you want to share temporarily (useful in Colab env)
  --gen_config_path GEN_CONFIG_PATH
                        which config to use for GenerationConfig
```

## Design figure

<p align="center">
  <img src="https://i.ibb.co/w069GYg/Screenshot-2023-03-20-at-1-25-29-PM.png" />
</p>

## Acknowledgements

I am thankful to [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU resources to experiment with Alpaca-LoRA deployment and share it to communities to try out.
