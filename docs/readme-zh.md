# Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model —— 一个中文低资源的llama+lora方案

![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/camel.png)

鉴于[llama](https://github.com/facebookresearch/llama),[alpaca](https://github.com/tatsu-lab/stanford_alpaca),[guanaco](https://github.com/Guanaco-Model/Guanaco-Model.github.io)等羊驼模型的研发成功，我们希望基于LLaMA+instruction数据构建一个中文的羊驼模型，并帮助大家能快速学会使用引入自己的数据，并训练出属于自己的小羊驼（Vicuna）。

项目包括

- finetune模型的代码
- 推理的代码
- 仅使用CPU推理的代码 (使用C++) 

## What‘s New

- March 23, 2023：开放了使用50w条数据训练的checkpoint-4000
- March 23, 2023：在colab上部署了fine-tuning和inference的代码
- March 23, 2023：提供了使用纯C++在CPU上进行推理的方案
- March 24, 2023：开放了使用100w条数据训练的checkpoint-8000

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

**Checkpoint-4000**

| Human                                            | Chinese-Vicuna                                               |
| ------------------------------------------------ | ------------------------------------------------------------ |
| 生成一份python的排序代码                         | A：def sort_list(lst):<br/>    lst.sort()<br/>    return lst |
| 自然语言处理是什么                               | 自然语言处理是一种计算机科学领域，旨在使计算机能够理解和处理自然语言。 |
| 能给我讲一段笑话吗                               | 为什么猫吃鱼？因为它们喜欢游泳。                             |
| 将下面文字翻译成英文。猫喜欢吃鱼，所以它们会游泳 | Cats like to eat fish, so they will swim.                    |
| 233和122的和等于多少                             | 233和122的和等于355。                                        |
| 华为是一家怎样的公司                             | 华为是一家全球范围内最大的移动通信设备制造商之一，以及一家全球范围内最大的移动通信设备制造商之一。华为公司的业务涵盖包括移动通信设备、互联网设备、智能家居设备等。 |
| 我想和女朋友在北京约会，能推荐几个约会的地方吗   | 北京有很多约会的地方，可以推荐以下几个约会的地方：<br/><br/>1. 北京艺术博物馆<br/>2. 颐和园<br/>3. 北京故宫博物馆<br/>4. 北京天安门广场<br/>5. 颐和园风景区 |
| 用python写一个二分查找的代码                     | ```python<br/>def binary_search(arr, target):<br/>    low = 0<br/>    high = len(arr) - 1<br/>    while low <= high:<br/>        mid = (low + high) // 2<br/>        if arr[mid] == target:<br/>            return mid<br/>        elif arr[mid] < target:<br/>            low = mid + 1<br/>        else:<br/>            high = mid - 1<br/>    return -1 |
| 怎么让自己精力充沛，列5点建议                    | 1. 保持良好的睡眠习惯；2. 定期锻炼身体；3. 学习新技能或知识；4. 与家人和朋友交流；5. 多听放松音乐。 |
| 父母都姓吴，取一些男宝宝和女宝宝的名字           | 男宝宝可以是吴杰、吴伟、吴飞、吴荣等。女宝宝可以是吴华、吴秀、吴美、吴婷等。 |

- 这个lora是我们用一张2080训练了30多个小时得到的结果（checkpoint-4000），大概50w条数据

**Checkpoint-8000**

| Human                                                        | Chinese-Vicuna                                               | Note                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- |
| 写一篇英文散文诗，主题是春雨，想象自己是春雨，和英国古代诗人莎士比亚交流 | A: Spring rain falls softly on my face,<br/>A cool breeze blows through my hair,<br/>The sun shines brightly in the sky,<br/>And I feel as if I am flying high.<br/><br/>The birds chirp happily in the trees,<br/>The flowers bloom and their scent fills the air,<br/>The bees buzz gently from flower to flower,<br/>And the world seems to come alive once more.<br/><br/>I can hear the rush of the river,<br/>The sound of the waterfalls,<br/>The splash of the raindrops on the rocks,<br/>And the gentle murmur of the stream.<br/><br/>My heart is filled with joy and peace,<br/>As I take in the beauty around me,<br/>The warmth of the sun on my skin,<br/>And the freshness of the spring air. | Repetition Penalty=2.2  |
| 写一篇英文散文诗，主题是春雨，想象自己是春雨，和英国古代诗人莎士比亚交流 | 春雨滴滴滚滚，<br/>滚滚滚滚滚滚。<br/>滚滚滚滚滚滚，<br/>滚滚滚滚滚滚。<br/><br/>春雨滚滚滚，<br/>滚滚滚滚滚滚。<br/>滚滚滚滚滚滚，<br/>滚滚滚滚滚滚。<br/><br/>春雨滚滚滚，<br/>滚滚滚滚滚滚。<br/>滚滚滚滚滚滚，<br/>滚滚滚滚滚滚。<br/><br/>春雨滚滚滚，<br/>滚滚滚 | Repetition Penalty=1.0  |
| Spring rain falls softly on my face,<br/>A cool breeze blows through my hair,<br/>The sun shines brightly in the sky,<br/>And I feel as if I am flying high.<br/><br/>The birds chirp happily in the trees,<br/>The flowers bloom and their scent fills the air,<br/>The bees buzz gently from flower to flower,<br/>And the world seems to come alive once more.<br/><br/>I can hear the rush of the river,<br/>The sound of the waterfalls,<br/>The splash of the raindrops on the rocks,<br/>And the gentle murmur of the stream.<br/><br/>My heart is filled with joy and peace,<br/>As I take in the beauty around me,<br/>The warmth of the sun on my skin,<br/>And the freshness of the spring air.<br/>把上面的散文诗翻译成中文 | 春雨轻轻洒落在我的脸上，<br/>轻轻的风轻轻吹过我的头发，<br/>阳光明媚地照耀在天空，<br/>我感觉自己在飞高。<br/><br/>鸟儿在树枝上唱歌，<br/>花朵开放，它们的香气充满了空气，<br/>蜜蜂在花朵间飞舞，<br/>它们的鸣声充满了温暖。<br/><br/>我听到河流的声音，<br/>水瀑在岩石上轻轻滴滴，<br/>雨水在石头上轻轻滴滴，<br/>它们的声音充满了宁静。<br/><br/>我的心充满了喜悦和平，<br/>我感受到热浪的阳光，<br/>温暖的空气洒在我的脸上，<br/>我感受到春季的美 | beams=2，max tokens=350 |
| Classify the sentiment of the following sentence into Positive, Neutral, or Negative:<br/>Super excited about teaching Stanford’s first course on Large Language Models! Check the syllabus out here | Positive                                                     |                         |
| 今天天气怎么样，把这句话翻译成英语                           | What's the weather like today?                               |                         |
| 今天天气怎么样，把这句话翻译成日语                           | 今天の天気は何ですか？                                       |                         |
| 明天就假期结束了，有点抗拒上班，应该什么办？                 | 明天就假期结束了，有点抗拒上班，应该什么办？可以考虑一些有趣的活动，比如去看电影、阅读书籍、听音乐、去游泳池或者参加户外活动等。此外，也可以考虑一些家庭活动，比如做烹饪、做玩具、做手工艺品等。 |                         |
| Tell me five words that rhyme with 'shock'.                  | Five words that rhyme with'shock' are: <br/>1. Rock <br/>2. Lock <br/>3. Cock <br/>4. Hock <br/>5. Stock |                         |

- 我们可以发现你不惩罚一下他，他就乱来，如果生成效果不好，可以增大Repetition Penalty、beams、topk
- 如果你的输入比较长，建议可以把beams调小
- 目前模型有一定的日语能力，但是不强

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

  - 我们提供了一个在上面混合数据上训练的lora模型
    - lora model
      -  https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-4000  
      - https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-8000  
    - 由于比较小暂时直接传在github上，后续会将更多的lora模型传在huggingface或网盘上
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

本地的python环境是3.8，torch是1.13.1，CUDA是12，transformers是4.28.0.dev0，tokenizers是0.13.2，sentencepiece是0.1.97

**多卡训练**

```bash
bash finetune.sh
```

- 这里需要注意的参数如下
  - TOT_CUDA，填写需要使用的GPU编号，如`TOT_CUDA="0,1,2,3"`
  - PORT，填写对应的端口
  - DATA_PATH，填写对应的数据位置，格式为json
  - OUTPUT_PATH，保存模型的相对路径
  - MODEL_PATH，上游模型
  - wandb：这是一个训练可视化工具，脚本中默认没开，可以在脚本中加入"--wandb"来开启

**单卡训练**

```
python finetune.py --data_path merge.json --test_size 2000
```

- 这个test_size不能大于数据大小

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

## **使用纯C++在CPU上进行推理**

详情见`tools`的[readme](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/readme.md)


## **更多工具**

我们还提供了:
- 其他模型权重的下载方式 ( 更快， 8MB/s ) : [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/download_llama.sh)
- 格式转换工具：llama系列模型参数文件的facebook格式 (`consolidated.xx.pth`) 和huggingface格式 (`pytorch_model-000xx-of-000xx.bin`): [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/convert_llama.py)


# todo

- [x] belle+guanaco(0.72 epoch, 4000 step)
- [ ] belle+guanaco(100%)
- [ ] 加入更多类似chitchat的对话型语料，增强自由对话的能力
- [x] 增加colab训练+lora载入接口
- [ ] Add the interaction capabilities
- [x] 增加llama的c++推理

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
