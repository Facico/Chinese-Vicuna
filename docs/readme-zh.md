# Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model —— 一个中文低资源的llama+lora方案

![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/camel.png)

鉴于[llama](https://github.com/facebookresearch/llama),[alpaca](https://github.com/tatsu-lab/stanford_alpaca),[guanaco](https://github.com/Guanaco-Model/Guanaco-Model.github.io)等羊驼模型的研发成功，我们希望基于LLaMA+instruction数据构建一个中文的羊驼模型，并帮助大家能快速学会使用引入自己的数据，并训练出属于自己的小羊驼（Vicuna）。

项目包括

- finetune模型的代码
- 推理的代码
- 仅使用CPU推理的代码 (使用C++) 

## 概述

相关技术

-  LLaMA paper: https://arxiv.org/abs/2302.13971v1
-  Self-Instruct paper: https://arxiv.org/abs/2212.10560
-  data generation: https://github.com/LianjiaTech/BELLE and https://guanaco-model.github.io/
-  the first work: https://github.com/tatsu-lab/stanford_alpaca

我们目前选择BELLE和Guanaco数据的组合作为我们的主要训练数据集。我们还将增加更多的闲聊数据集（例如[LCCC](https://github.com/thu-coai/CDial-GPT)）来支持闲聊对话。

## 意义在哪

类似于stable diffusion模型的爆火，出现了像civitai等平台，由一个基础的模型+各种LORA模型的开源社区。

本项目希望帮助大家去训练这个LORA

- 什么是LORA
  - 简单的说就是用来帮大模型适应你的数据集的一个插件，技术细节见[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)，他的优点是finetune的时候非常的快，得到的模型也很小，大概30M左右，关键是支持**即插即用**。可以预见，这是一个非常适合开源生态的架构。

我们这里，将通过非常低配置的环境，帮助大家训练，仅一张**2080**（11G）就能取得一定的效果。

## 在colab上快速部署

| colab link                                                   | Descriptions                                           |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1OLCJ-ZHogm5O3RdyUDY83YfgnCXdHNXp?usp=sharing) | 加载llama7B和对应的lora模型推理，并提供一个简单的webui |
| [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1SOOJjxp02uuUFDNgm4jkvaA_IdNnnvdk?usp=sharing) | 使用我们收集的数据微调                                 |

## 模型效果

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
​```
Q：怎么让自己精力充沛，列5点建议
A：1. 保持良好的睡眠习惯；2. 定期锻炼身体；3. 学习新技能或知识；4. 与家人和朋友交流；5. 多听放松音乐。
Q：父母都姓吴，取一些男宝宝和女宝宝的名字
A：男宝宝可以是吴杰、吴伟、吴飞、吴荣等。女宝宝可以是吴华、吴秀、吴美、吴婷等。
```

由于目前训练轮数不多，在自由对话上效果还一般，但我们可以看到已经具备了不错的能力

- 这个lora是我们用一张2080训练了30个小时得到的结果（checkpoint-3000），大概40wstep

## 训练一个lora需要什么

- 代码：

  - 此代码基于alpaca-lora开发，https://github.com/tloen/alpaca-lora
  - 这是一套比较简单的代码，基本思路就是用[PEFT](https://github.com/huggingface/peft)的lora接口+transformer的trainer+instruction的数据配置

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

  - 目前我们整合的数据可以在百度网盘或google drive上下载

    - 链接: https://pan.baidu.com/s/1WSxuhSAotl14ifaAiz5eKw?pwd=b4kb 提取码: b4kb 
    - 链接: https://drive.google.com/file/d/1tzXVhS74m-EtoFot7hEc005LDeZGPit_/view?usp=sharing

- 上游模型：

  - LLAMA 7B（当然，如果你有更大的机器可以换成13B的，LLAMA13B在数值上优于175B的GPT3）

- lora模型：

  - 我们提供了一个在上面混合数据上训练了大概40w条的lora模型，见 https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-3000  
    - 由于比较小暂时直接传在github上
    - 模型使用的是8bit+lora+256 tokens

- 设备：

  - 训练：一张2080Ti即可。由于数据长度都在256以内，大概占用9G显存。
    - 70w的数据，3个epoch，一张2080Ti大概200h
  - 推理：一张2080Ti即可。
  - 我们对纯CPU上推理也进行了支持，详情见[`tools`](https://github.com/Facico/Chinese-Vicuna/blob/master/tools)
  

## 怎么使用

**安装**

```
pip install -r requirements.txt
```

本地的python环境是3.8，torch是1.13.1，CUDA是12

**训练**

```bash
bash finetune.sh
```

- 这里需要注意的参数如下
  - TOT_CUDA，填写需要使用的GPU编号，如`TOT_CUDA="0,1,2,3"`
  - PORT，填写对应的端口
  - DATA_PATH，填写对应的数据位置，格式为json
  - OUTPUT_PATH，保存模型的相对路径
  - MODEL_PATH，上游模型

**inference并使用gradio生成一个网页**

```bash
bash generate.sh
```

- 这里需要注意的参数如下
  - BASE_MODEL，上游模型
  - LORA_PATH，lora模型的checkpoint文件夹
    - 这里要注意的是，lora模型加载的config必须是"adapter_config.json"，模型名字必须是“adapter_model.bin”，不过在训练的时候会自动保存为“pytorch_model.bin”，而"adapter_config.json"和“adapter_model.bin”会在全部训练结束之后保存
      - 如果你是在训练的checkpoint中载入的lora模型，代码里会自动帮你把本地的"config-sample/adapter_config.json"复制到对应目录，并把“pytorch_model.bin”改名为“adapter_model.bin”

- 使用的时候，"max_tokens"根据自己电脑的显存来设置，如果生成的内容产生了很多重复信息，可以将"Repetition Penalty"调高

**仅使用CPU进行推理**

详情见`tools`的[readme](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/readme.md)

# todo

- [x] belle+guanaco(0.72 epoch, 4000 step)
- [ ] belle+guanaco(100%)
- [ ] 加入更多类似chitchat的对话型语料，增强自由对话的能力
- [x] 增加colab训练+lora载入接口
- [ ] 增加llama的c++推理

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
