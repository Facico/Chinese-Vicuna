![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/vicuna-llama.png)

# Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model —— 一个中文低资源的llama+lora方案

 | [English](https://github.com/Facico/Chinese-Vicuna/blob/master/README.md) | [中文](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md) |

![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/camel.png)

This is the repo for the Chinese-Vicuna project, which aims to build and share an instruction-following Chinese LLaMA model which can run on a single Nvidia RTX-2080TI, that why we named this project `Vicuna`, small but strong enough ! 

- Why is it called Vicuna：In view of the successful development of alpaca models such as [llama](https://github.com/facebookresearch/llama),[alpaca](https://github.com/tatsu-lab/stanford_alpaca),[guanaco](https://github.com/Guanaco-Model/Guanaco-Model.github.io)，We want to train a Chinese small alpaca like Vicuna.

The repo contains:

- code for finetune the model ()
- code for generation based on trained model

## Overview

-  LLaMA paper: https://arxiv.org/abs/2302.13971v1
-  Self-Instruct paper: https://arxiv.org/abs/2212.10560
-  data generation: https://github.com/LianjiaTech/BELLE and https://guanaco-model.github.io/
-  the first work: https://github.com/tatsu-lab/stanford_alpaca

We currently select the combination of BELLE and Guanaco data as our main training dataset. We will also add more chitchat dataset ( e.g. [LCCC](https://github.com/thu-coai/CDial-GPT) ) to support casual conversation.

## What is the meaning?

Similar to the explosion of the stable diffusion model, platforms like civitai have emerged, consisting of a base model + various LORA models in an open source community.

The repo hopes to help you to train these LORA models.

**what is LORA?**：Simply, it's a plugin used to help adapt large models to your dataset, technical details can be found in[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)。Its advantage is that the finetune is very fast when it comes to getting a small model, about 30M, and the key is that it supports **Plug and Play**. As can be expected, this is a very suitable architecture for open source ecology.

Here, we will help you train through a very low-demand environment, with only one **2080** (11G) to achieve certain results.

## Try on colab

| colab link                                                   | Descriptions                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1OLCJ-ZHogm5O3RdyUDY83YfgnCXdHNXp?usp=sharing) | Use the specified lora model to inference and generate a webui |
| [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1SOOJjxp02uuUFDNgm4jkvaA_IdNnnvdk?usp=sharing) | Finetune with the data we collected                          |

## Performance

```
Q：生成一份python的排序代码
A：def sort_list(lst):
    lst.sort()
    return lst
Q：自然语言处理是什么
A：自然语言处理是一种计算机科学领域，旨在使计算机能够理解和处理自然语言。
Q：能给我讲一段笑话吗
A：为什么猫吃鱼？因为它们喜欢游泳。
Q：将下面文字翻译成英文。猫喜欢吃鱼，所以它们会游泳
A：Cats like to eat fish, so they will swim.
Q：233和122的和等于多少
A：233和122的和等于355。
Q：华为是一家怎样的公司
A：华为是一家全球范围内最大的移动通信设备制造商之一，以及一家全球范围内最大的移动通信设备制造商之一。华为公司的业务涵盖包括移动通信设备、互联网设备、智能家居设备等。
Q：我想和女朋友在北京约会，能推荐几个约会的地方吗
A：北京有很多约会的地方，可以推荐以下几个约会的地方：

1. 北京艺术博物馆
2. 颐和园
3. 北京故宫博物馆
4. 北京天安门广场
5. 颐和园风景区
Q：用python写一个二分查找的代码
A：
​```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
Q：怎么让自己精力充沛，列5点建议
A：1. 保持良好的睡眠习惯；2. 定期锻炼身体；3. 学习新技能或知识；4. 与家人和朋友交流；5. 多听放松音乐。
Q：父母都姓吴，取一些男宝宝和女宝宝的名字
A：男宝宝可以是吴杰、吴伟、吴飞、吴荣等。女宝宝可以是吴华、吴秀、吴美、吴婷等。
```

Due to the small number of training rounds so far, the effect on casual conversations is still average, but we can see that there is already a good ability. This LORA model is the result of 30 hours of training with a 2080Ti (checkpoint-3000), about 40wstep

## What we need?

- code：

  - This code is developed based on alpaca-lora，https://github.com/tloen/alpaca-lora
  - This is a relatively simple set of code, the basic idea is to use PEFT's lora interface + transformer's trainer + instruction data configuration

- data：

  - We have utilized several current high quality open source data and are very grateful for their contributions. Many of these data use chatgpt's interface like alpaca to generate high quality INSTRUCTION data.

    - [Belle](https://github.com/LianjiaTech/BELLE)
    - [guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

  - The data format is relatively simple, basically as follows, with simple examples such as：[`./sample/merge_sample.json`](https://github.com/Facico/Chinese-Vicuna/blob/master/sample/merge_sample.json)

    - ```
      {
      'instruction': 
      'input': 
      'output'
      }
      ```

    - That is, an instruction, an input, and an output are required. since the data is processed by directly linking instruction and input, the data can actually require only instruction and output, as

      ```
       {
        'instruction': "用一句话描述地球为什么是独一无二的。\\n\n"
        'input': ""
        'output': "地球上有适宜生命存在的条件和多样化的生命形式。"
        }
      ```

      

  - The data we currently integrate is available for download on BaiduDownload or Google Drive

    - link: https://pan.baidu.com/s/1WSxuhSAotl14ifaAiz5eKw?pwd=b4kb   password: b4kb 
    - link: https://drive.google.com/file/d/1tzXVhS74m-EtoFot7hEc005LDeZGPit_/view?usp=sharing

- Large Language Model：

  - LLAMA 7B（Of course, if you have a larger machine(such as 3090Ti) can be replaced with a 13B, LLAMA13B is numerically superior to 175B GPT3）

- LORA model：

  - We provide a lora model trained on the above mixed data with roughly 50w entries, see https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-4000  
    - Since the model is relatively small, it is temporarily uploaded on github(about 30M)
    - The model uses 8bit+lora+256 tokens

- Device：

  - Training：A 2080Ti is sufficient. Since the data length is within 256, it takes about 9G of video memory.
    - 70w of data, 3 epochs, a 2080Ti about 200h
  - Inference：A 2080Ti is all you need。

  

## How to use

**Installation**

```
git clone https://github.com/Facico/Chinese-Vicuna
pip install -r requirements.txt
```

Local python environment is 3.8, torch is 1.13.1, CUDA is 12

**Training**

```bash
bash finetune.sh
```

- The parameters to note here are as follows
  - TOT_CUDA, fill in the GPU number to be used, such as `TOT_CUDA="0,1,2,3"`
  - PORT, fill in the corresponding port
  - DATA_PATH，fill in the corresponding data location in the format of json
  - OUTPUT_PATH，fill in the relative path to save the model
  - MODEL_PATH，path of LLM

**inference and use gradio to generate a web page**

```bash
bash generate.sh
```

- The parameters to note here are as follows

  - BASE_MODEL，path of LLM
  - LORA_PATH，The checkpoint folder of the lora model
    - It should be noted here that the config loaded by the lora model must be "adapter_config.json" and the model name must be "adapter_model.bin", but it will be automatically saved as "pytorch_model.bin" during training. pytorch_model.bin" during training, while "adapter_config.json" and "adapter_model.bin" will be saved after all training is finished
      - If you load the lora model in the training checkpoint, the code will automatically copy the local "config-sample/adapter_config.json" to the corresponding directory for you and rename the "pytorch_model.bin" to "adapter_model.bin". and rename "pytorch_model.bin" to "adapter_model.bin".

- When using, "max_tokens" is set according to your computer's video memory, and if the generated content generates a lot of duplicate information, you can turn up the "Repetition Penalty".

# todo

- [x] belle+guanaco(0.72 epoch, 4000 step)
- [ ] belle+guanaco(100%)
- [ ] Add more chitchat-like conversational corpus to enhance free conversation
- [x] Add colab training + lora loading interface
- [ ] Add llama c++ inference

# Citation

If you find this project useful in your research, please consider citing:

```
@inproceedings{leng2023chinese-vicuna,
  title={Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model},
  author={Chenghao Fan, Zhenyi Lu and Jie Tian},
  url={https://github.com/Facico/Chinese-Vicuna},
  year={2023}
}
```
