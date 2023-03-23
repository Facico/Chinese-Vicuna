# Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model

![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/camel.png)

鉴于[llama](https://github.com/facebookresearch/llama),[alpaca](https://github.com/tatsu-lab/stanford_alpaca),[guanaco](https://github.com/Guanaco-Model/Guanaco-Model.github.io)等羊驼模型的研发成功，我们希望构建一个中文的羊驼模型，并帮助大家能快速学会使用引入自己的数据，并训练出属于自己的羊驼。



This is the repo for the Chinese-Vicuna project, which aims to build and share an instruction-following Chinese LLaMA model which can run on a single Nvidia RTX-2080TI, that why we named this project `Vicuna`, small but strong enough !

The repo contains:

- code for finetune the model ()
- code for generation based on trained model

## Overview

-  LLaMA paper: https://arxiv.org/abs/2302.13971v1
-  Self-Instruct paper: https://arxiv.org/abs/2212.10560
-  data generation: https://github.com/LianjiaTech/BELLE and https://guanaco-model.github.io/
-  the first work: https://github.com/tatsu-lab/stanford_alpaca

We currently select the combination of BELLE and Guanaco data as our main training dataset. We will also add more chitchat dataset ( e.g. [LCCC](https://github.com/thu-coai/CDial-GPT) ) to support casual conversation.

## 意义在哪

类似于stable diffusion模型的爆火，出现了像civitai等平台，由一个基础的模型+各种LORA模型的开源社区。

本项目希望帮助大家去训练这个LORA

- 什么是LORA
  - 简单的说就是用来帮大模型适应你的数据集的一个插件，技术细节见[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)，他的优点是finetune的时候非常的快，得到的模型也很小，大概30M左右，关键是支持**即插即用**。可以预见，这是一个非常适合开源生态的架构。

我们这里，将通过非常低配置的环境，帮助大家训练，仅一张**2080**（11G）就能取得一定的效果。

## What we need

- 代码：
  - 此代码基于alpaca-lora开发，https://github.com/tloen/alpaca-lora
  - 这是一套比较简单的代码，基本思路就是用[PEFT](https://github.com/huggingface/peft)的lora接口+transformer的trainer+instruction的数据配置
- 上游模型：
  - LLAMA 7B（当然，如果你有更大的机器可以换成13B的，LLAMA13B在数值上优于175B的GPT3）

- 设备：
  - 训练：一张2080Ti即可。由于数据长度都在256以内，大概占用9G显存。
  - 推理：一张2080Ti即可。

- 数据：

  - 我们利用了目前几份高质量的开源数据，非常感谢他们的贡献。这些数据很多都像alpaca那样，使用chatgpt的接口，生成高质量的instruction数据。
    - [Belle](https://github.com/LianjiaTech/BELLE)
    - [guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

  - 数据格式比较简单，基本如下，简单的例子如：[`./sample/merge_sample.json`](https://github.com/Facico/Chinese-Vicuna/blob/master/sample/merge_sample.json)

    - ```
      {
      'instruction': 
      'input': 
      'output'
      }
      ```

    - 即需要一个指令，一个input，一个output。由于数据处理的时候是直接将instruction和input连接起来的，所以数据其实可以只需要instruction和output，如

      - ```
        {
        'instruction': "用一句话描述地球为什么是独一无二的。\\n\n"
        'input': ""
        'output': "地球上有适宜生命存在的条件和多样化的生命形式。"
        }
        ```

  - 整理的数据后续我们将上传上来

## How to use

训练

```bash
bash finetune.sh
```

- 这里需要注意的是几个参数
  - TOT_CUDA，填写需要使用的GPU编号，如`TOT_CUDA="0,1,2,3"`
  - PORT，填写对应的端口
  - DATA_PATH，填写对应的数据位置，格式为json
  - OUTPUT_PATH，保存模型的相对路径
  - MODEL_PATH，上游模型

inference并使用gradio生成一个网页

```bash
bash generate.sh
```



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

