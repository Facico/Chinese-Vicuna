![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/vicuna-llama.png)
# Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model —— 一个中文低资源的llama+lora方案
![GitHub Repo stars](https://img.shields.io/github/stars/Facico/Chinese-Vicuna?style=social) [![HuggingFace badge](https://camo.githubusercontent.com/4a295d6d34ed2c79cfe624ce6358a4be53d4187c883aaa9345fdc322937ce542/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393748756767696e67466163652d4a6f696e2d79656c6c6f77)](https://huggingface.co/Chinese-Vicuna)
 
 | [English](https://github.com/Facico/Chinese-Vicuna/blob/master/README.md) | [中文](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md) |

![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/camel.png)

鉴于[llama](https://github.com/facebookresearch/llama),[alpaca](https://github.com/tatsu-lab/stanford_alpaca),[guanaco](https://github.com/Guanaco-Model/Guanaco-Model.github.io)等羊驼模型的研发成功，我们希望基于LLaMA+instruction数据构建一个中文的羊驼模型，并帮助大家能快速学会使用引入自己的数据，并训练出属于自己的小羊驼（Vicuna）。

项目包括

- finetune模型的代码
- 推理的代码
- 仅使用CPU推理的代码 (使用C++) 
- 下载/转换/量化Facebook llama.ckpt的工具
- 其他应用

这里分别是我们单轮和多轮的问答效果 (由于默认设置 beam-size=4, 所以视频里边会看到 4 个打印进程同时输出):

https://user-images.githubusercontent.com/72137647/228496412-60043912-f491-430b-848a-599e6edfa5ef.mp4

https://user-images.githubusercontent.com/72137647/229739363-1b48f3a9-02a1-46ab-81ee-8c62dc1399b2.mp4

## What‘s New

- March 23, 2023：开放了在belle+guanaco数据上训练50w条数据训练的checkpoint-4000
- March 23, 2023：在colab上部署了fine-tuning和inference的代码
- March 23, 2023：提供了使用纯C++在CPU上进行推理的方案
- March 24, 2023：开放了在belle+guanaco数据上训练1.5个epoch的checkpoint-final（ 大约100w条）
- March 26, 2023：提供了LLaMA模型的量化方法
- March 27, 2023：开放了在belle+guanaco数据上训练3个epoch的checkpoint-final
- March 27, 2023：增加了多轮交互式对话脚本与alpaca-lora-serve服务
- March 28, 2023：在[huggingface](https://huggingface.co/Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco)上开放了我们的模型
- March 29, 2023：我们对gradio-UI改进，添加了更好的用户支持(支持beam search的打字机输出效果，清除对话历史，重置参数)
- March 29, 2023：增加了断点重训接口，支持从我们的checkpoint继续训练其他数据集
- March 29, 2023: 开放了我们训练的13B lora模型[13B-based lora model](https://huggingface.co/Chinese-Vicuna)
- March 29, 2023：增加了更详细的[performance样例](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance.md)
- April 1, 2023: 在`chat.py`对多轮对话提供了更好的支持：( 支持4种生成模式的流式输出/打字机效果: beam search, greedy, sample, beam sample ; 我们还提供了取消当前对话的功能 )
- April 4, 2023: 增加了13B的[performance样例](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-13B.md)

**相关技术**

-  LLaMA paper: https://arxiv.org/abs/2302.13971v1
-  Self-Instruct paper: https://arxiv.org/abs/2212.10560
-  data generation: https://github.com/LianjiaTech/BELLE and https://guanaco-model.github.io/
-  the first work: https://github.com/tatsu-lab/stanford_alpaca

我们目前选择BELLE和Guanaco数据的组合作为我们的主要训练数据集。
我们还将增加多轮instruction数据增强上下文能力。

## 目录

- [Vicuna](https://github.com/Facico/Chinese-Vicuna)
  - [新的进展](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#whats-new)
  - [意义在哪](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#%E6%84%8F%E4%B9%89%E5%9C%A8%E5%93%AA)
  - [在colab上快速部署](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#%E5%9C%A8colab%E4%B8%8A%E5%BF%AB%E9%80%9F%E9%83%A8%E7%BD%B2)
  - [模型效果](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#%E6%A8%A1%E5%9E%8B%E6%95%88%E6%9E%9C)
    - **Checkpoint-4000**(Facico/Chinese-Vicuna-lora-7b-0.75epoch-belle-and-guanaco)
    - **Checkpoint-8000**(Facico/Chinese-Vicuna-lora-7b-1.5epoch-belle-and-guanaco)
    - **Checkpoint-final**(Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco)和它用来多轮对话
  - [训练一个lora需要什么](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AAlora%E9%9C%80%E8%A6%81%E4%BB%80%E4%B9%88)
    - 代码、数据、上游模型、lora模型、设备
  - [怎么使用](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#%E6%80%8E%E4%B9%88%E4%BD%BF%E7%94%A8)
    - 安装、多卡训练、单卡训练、推理并生成一个webui(支持流式+beam search)、多轮交互并生成一个webui(支持流式+beam search)、基于alpaca-lora-serve的流式交互(不支持beam search)
  - [使用纯C++在CPU上推理](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#%E4%BD%BF%E7%94%A8%E7%BA%AFc%E5%9C%A8cpu%E4%B8%8A%E8%BF%9B%E8%A1%8C%E6%8E%A8%E7%90%86)
  - [更多工具](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#%E6%9B%B4%E5%A4%9A%E5%B7%A5%E5%85%B7)，详见[tool readme](https://github.com/Facico/Chinese-Vicuna/tree/master/tools)
    - 其他模型权重的快速下载工具`download_llama.sh`
    - llama系列模型，facebook格式和huggingface格式转换工具`convert_llama.py`
    - 使用gptq的llama量化工具
  - [可能遇到的问题](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#%E5%8F%AF%E8%83%BD%E4%BC%9A%E9%81%87%E5%88%B0%E7%9A%84%E9%97%AE%E9%A2%98)
  - [todo](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#todo)
  - [citation](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md#citation)

## 意义在哪

类似于stable diffusion模型的爆火，出现了像civitai等平台，由一个基础的模型+各种LORA模型的开源社区。

本项目希望帮助大家去训练这个LORA

- **什么是LORA**
  - 简单的说就是用来帮大模型适应你的数据集的一个插件，技术细节见[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)，他的优点是finetune的时候非常的快，得到的模型也很小，大概30M左右，关键是支持**即插即用**。可以预见，这是一个非常适合开源生态的架构。

我们这里，将通过非常低配置的环境，帮助大家训练，仅一张**2080**（11G）就能取得一定的效果。

## 在colab上快速部署

| colab link                                                   | Descriptions                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1OLCJ-ZHogm5O3RdyUDY83YfgnCXdHNXp?usp=sharing) | 加载llama7B和对应的lora模型推理，并提供一个简单的webui，支持beam search流式输出 |
| [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1SOOJjxp02uuUFDNgm4jkvaA_IdNnnvdk?usp=sharing) | 使用我们收集的数据微调                                       |
| [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1Yc38cKifY1_sfTBJpEEKin8m2rxoYXtm?usp=sharing) | 为多轮对话交互接口提供一个webui                              |

## 模型效果

7B的模型效果参见[这里](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance.md), 13B的模型效果参见[这里](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-13B.md)。也有伙伴用我们的方法在33B模型上取得了不错的效果。

历史checkpoint记录参见[这里](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-history.md)


<details>
<summary><b>checkpoint-4000(7B)</b></summary>

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

- This LORA model is the result of 30 hours of training with a 2080Ti (checkpoint-4000), about 50w data

</details>

<details>
<summary><b>checkpoint-8000(7B, about 1.5 epoch on belle+guanaco)</b></summary>

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

- We can find that you do not punish him a little, he will mess up, if the generation effect is not good, you can increase the Repetition Penalty, beams, topk.
- If your input is long, we suggest you can turn down the beams
- The checkpoint-8000 has some Japanese language skills, but not strong

</details>

<details>
<summary><b>checkpoint-final(7B, 3 epoch on belle+guanaco)</b></summary>


| Human                                                        | Chinese-Vicuna                                               | Note                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------- |
| 生成一份python的排序代码                                     | def bubble_sort(arr):<br/>    for i in range(len(arr)):<br/>        for j in range(len(arr)-i-1):<br/>            if arr[j] > arr[j+1]:<br/>                arr[j], arr[j+1] = arr[j+1], arr[j]<br/>    return arr |                        |
| 生成一份简单的python的排序代码                               | def sort_list(lst):<br/>    return sorted(lst)               |                        |
| 用python写一个二分查找的代码                                 | def binary_search(arr, target):<br/>    low = 0<br/>    high = len(arr) - 1<br/>    while low <= high:<br/>        mid = (low + high) // 2<br/>        if arr[mid] == target:<br/>            return mid<br/>        elif arr[mid] < target:<br/>            low = mid + 1<br/>        else:<br/>            high = mid - 1<br/>    return -1 |                        |
| 用python写一个递归的二分查找的代码                           | def binary_search(arr, target):<br/>    if len(arr) == 0:<br/>        return -1<br/>    mid = len(arr) // 2<br/>    if arr[mid] == target:<br/>        return mid<br/>    if arr[mid] < target:<br/>        return binary_search(arr, target, 0, mid-1)<br/>    else:<br/>        return binary_search(arr, target, mid+1, len(arr)-1) |                        |
| 自然语言处理是什么                                           | 自然语言处理（Natural Language Processing，NLP）是一种计算机科学领域，旨在使计算机能够理解、处理和生成自然语言。它包括机器翻译、语音识别、文本分类、情感分析等技术。 |                        |
| 能给我讲一段笑话吗                                           | 为什么鸟儿喜欢唱歌？因为它们知道，如果不唱歌，他们就会被鸟巢抓住！ | Repetition Penalty=2.0 |
| 能给我讲一段关于狗笑话吗                                     | 为什么狗不喜欢骑自行车？因为它们总是想要下车！               | Repetition Penalty=2.0 |
| 将下面文字翻译成英文。猫喜欢吃鱼，所以它们会游泳             | Cats like to eat fish, so they will swim.                    |                        |
| 我想和女朋友在北京约会，能推荐几个约会的地方吗               | 北京有很多很好的约会地方，比如颐和园、故宫、天安门广场等。   |                        |
| 我想和女朋友在北京约会，能推荐几个约会的地方并给出攻略吗     | 北京有很多受欢迎的约会地方，以下是几个推荐：<br/><br/>1. 颐和园：颐和园是北京最著名的公园之一，拥有丰富的文化遗产和美丽的景观。<br/><br/>2. 天安门广场：天安门广场是北京最受欢迎的旅游景点之一，拥有丰富的文化遗产和美丽的景观。<br/><br/>3. 故宫博物院：故宫博物院是中国最受欢迎的旅游景点之一，拥有丰富的文化遗产和美丽的景观。<br/><br/>4. 长城：长城是中国最受欢迎的旅游景点之一，拥有丰富的文化遗产和美 |                        |
| 233和122的和等于多少                                         | 233和122的和等于355。                                        |                        |
| 华为是一家怎样的公司                                         | 华为是一家全球领先的技术公司，拥有丰富的技术研发能力和优质的产品线。它的产品涵盖了各种领域，包括移动通信、计算机、互联网、智能家居等。华为拥有丰富的研发人员和技术团队，以及优质的产品质量。 |                        |
| 西瓜是什么，请用详细的语言回答我                             | 西瓜是一种常见的水果，它有橙色、黄色和红色三种不同的外观。它的口感鲜美，味道清爽，富含多种维生素和矿物质，有助于保持身体健康。 |                        |
| 企鹅会飞吗？                                                 | 是的，企鹅可以飞行，它们的飞行速度可以达到100英里/小时。     |                        |
| 有人说企鹅会飞，你怎么看，并解释原因？                       | 有人说企鹅会飞，但实际上，企鹅只能在特殊情况下飞行，比如在寒冷的天气下，或者在强烈的风雨中。这是因为企鹅的飞行能力很低，而且它们的飞行速度也很慢，所以在寒冷的天气或者强烈的风雨中，企鹅可以避免被暴风雨摧毁。 |                        |
| 猪会飞吗？                                                   | 猪不会飞。                                                   |                        |
| 有人说猪会飞，你怎么看，并解释原因？                         | 很多人认为猪会飞，但实际上，猪不会飞。猪的飞行能力很低，它们只能在极端情况下飞行，比如受到强烈的风险或者受到威胁的情况下。猪的飞行能力很低，因为它们的肌肉比人类的肌肉更轻，而且它们的脊椎也比人类的脊椎更轻，所以它们的肌肉和脊椎都比人类的肌肉和脊椎更轻，因此它们的飞行能力也很低。 |                        |
| 最新研究显示，猪有飞行能力，你知道为什么吗？                 | 最新研究显示，猪有飞行能力是由于它们的骨骼结构，它们的骨骼结构比人类骨骼结构更加强壮，可以帮助它们更快地飞行。此外，猪的骨骼结构也可以帮助它们更快地跑步和跳跃。此外，猪的骨骼结构也可以帮助它们更快地跑步和跳跃。此外，猪的骨骼结构也可以帮助它们更快地飞行。此外，猪的骨骼结构也可以帮助它们更快地跑步和跳 |                        |
| 写一篇英文散文诗，主题是春雨，想象自己是春雨，和英国古代诗人莎士比亚交流 | Spring rain pours down from the sky,<br/>Washing away the dirt and grime.<br/>The birds chirp and sing in the trees,<br/>A symphony of joy and ease.<br/><br/>I am the spring rain, soaking up the sun,<br/>Cooling the earth and bringing new life.<br/>My droplets fall gently on the ground,<br/>Bringing hope and renewal to all.<br/><br/>I am the spring rain, washing away the pain,<br/>Clearing the way for new beginnings again.<br/>I am the spring rain, showering love and care,<br/>Giving strength and courage to those who dare. | Repetition Penalty=2.3 |
| Spring rain pours down from the sky,<br/>Washing away the dirt and grime.<br/>The birds chirp and sing in the trees,<br/>A symphony of joy and ease.<br/><br/>I am the spring rain, soaking up the sun,<br/>Cooling the earth and bringing new life.<br/>My droplets fall gently on the ground,<br/>Bringing hope and renewal to all.<br/><br/>I am the spring rain, washing away the pain,<br/>Clearing the way for new beginnings again.<br/>I am the spring rain, showering love and care,<br/>Giving strength and courage to those who dare.<br />将上述文字翻译成中文 | 春雨落下来，清洗灰尘和垃圾。<br/>鸟儿在树枝上唱歌，是一个美妙的声音。<br/>我是春雨，在阳光下洒落，<br/>清洗地球，带来新生命。<br/><br/>我是春雨，清洗痛苦和痛苦，<br/>让人们有希望和新开始。<br/>我是春雨，洒下爱和关心，<br/>给人们带来勇气和勇气。 |                        |
| 今天天气怎么样，把这句话翻译成日语                           | 今天の天気は何ですか？                                       | Repetition Penalty=3   |
| 将下面的话翻译成日语，中文小羊驼由各种高质量的开源instruction数据集，结合Alpaca-lora的代码训练而来 | 小羊驼は、各種高質量の開源instruction数据集を使用して、Alpaca-loraの代码によって訓練されました。 |                        |
| 怎么让自己精力充沛，列5点建议                                | 1. 保持良好的睡眠习惯，每晚充足的睡眠时间。<br/>2. 保持健康的饮食习惯，多吃新鲜水果和蔬菜，少吃高热量的食物。<br/>3. 坚持锻炼，每周至少进行30分钟的有氧运动，如散步、跑步、游泳等。<br/>4. 学习新知识，增强自己的知识和技能。<br/>5. 保持积极的心态，勇敢地面对困难，坚持自己的目标。 |                        |
| 中国足球要怎么进入世界杯，请给出详细的建议                   | 中国足球要怎么进入世界杯？首先，我们需要建立一个强大的足球团队，拥有优秀的球员和团队领袖。其次，我们需要建立一个强大的教练团队，拥有优秀的教练和团队领袖。第三，我们需要建立一个强大的训练基础，拥有优秀的训练设施和训练团队。第四，我们需要建立一个强大的资源基础，拥有优秀的资源支持和资金支持。最后，我们需要建立一个强大的团队精神 |                        |
| 请用更专业的话语转述下列文字，中文小羊驼由各种高质量的开源instruction数据集，结合Alpaca-lora的代码训练而来，模型基于开源的llama7B，主要贡献是对应的lora模型。由于代码训练资源要求较小，希望为llama中文lora社区做一份贡献。 | 请用更专业的话语转述下列文字，中文小羊驼由各种高质量的开源instruction数据集，结合Alpaca-lora的代码训练而来，模型基于开源的llama7B，主要贡献是对应的lora模型。由于代码训练资源要求较小，希望为llama中文lora社区做一份贡献。 |                        |

- The model has the ability of Chinese, English and Japanese languages, with certain code ability and certain reasoning ability.
- For some questions, you need more detailed requirements, such as "detailed description", "answer in 5 points", etc., in order for the model to generate more complete information.
- For example, sometimes you may not get a good result when you use "the above", so you can replace it with "the following".
- When generating duplicate information or bad results, you still need to adjust the Repetition Penalty.
  - Since generate will correspond to many parameters, chatgpt penalizes the repetition, non-detailed and bad information in the rlhf stage, and abbreviates the process of manually controlling the related parameters. Of course you can also directly set the Repetition Penalty to a larger size.

</details>

<details>
<summary><b>checkpoint-final(7B) 多功能测试</b></summary>


<details>
<summary><b>多轮对话、多轮翻译、角色扮演</b></summary>

**制作披萨**

```
> 你能不能详细介绍一下怎么做披萨？
当然可以！披萨是一种非常流行的意大利美食，通常由面团、番茄酱、奶酪和其他配料组成。首先，将面团放在平底锅上，加入番茄酱，再加入奶酪和其他配料，如洋葱、蘑菇、辣椒等。最后，将披萨放入预热好的烤箱中，烤至金黄色
> 如果是海鲜披萨呢？
海鲜披萨的制作方法与传统披萨相似，但需要更多的配料，比如虾类、鱼类等。首先，将面团放在平底锅上，加入番茄酱，再加入奶酪和其他配料，如洋葱、辣椒等。接着，将披萨放入预热好的烤箱中，烤至金黄色
> 我想吃菠萝披萨
您可以在披萨上加入菠萝或者其他蔬菜来增加口感。
```

**学生购买手机**

```
> 你推荐我买最新款的iPhone吗？
当然可以，这款iPhone是一款非常出色的智能手机。它拥有高性能的处理器、更大的存储空间和更长的电池续航时间。它还配备了全新的相机设备，使用者可以在任何情况下拍摄美丽的照片。
> 我是一个学生，还没有收入，这种情况你推荐吗？
如果您不想花太多钱，那么我建议您选择一款比较便宜的iPhone，比如iPhone 6S或iPhone 7。这些手机也具有很好的性能和功能，而且价格也比较便宜。
> 去哪里买这类产品？
您可以在Apple Store上购买这类产品，或者在其他电商平台上购买。
> 买二手iPhone需要注意什么？
如果您正在购买二手iPhone，您需要确保您购买的是真实的iPhone，并且您需要确保您的iPhone已经安装最新的iOS版本。此外，您还需要确保您的iPhone没有过过任何故障或破坏。
```

**介绍北京**

```
> 你是一个资深导游，你能介绍一下中国的首都吗
当然可以！中国的首都是北京。它是中国的政治、经济和文化中心，也是全球最大的城市之一。北京有着众多的历史遗迹，如故宫、天安门广场、长城等。此外，北京还有许多美丽的公园和博物馆，比如故宫博物院、清明公园、北京故宫博物馆等。
> 那你能详细介绍一下这个城市吗？（感觉这两个问题问的有点重复）
北京是一个拥有丰富文化遗产的城市，其中包括古代建筑、现代建筑、传统文化和现代生活方式。在北京，你可以欣赏到世界上最高的建筑——天安门广场，也可以参观清明公园，品尝当地美食，体验当地文化。此外，北京还有许多博物馆，比如故宫博物馆、北京故宫博物馆等。
> 还有其他推荐的景点吗？
您也可以去了北京的七朝塔、颐和园、颐和园博物馆等。
> 吃小吃的话推荐去哪里？
北京有许多好吃的小吃，比如烤鸭、火锅、炸酱面等等。您可以去北京西湖区的小吃街，或者去北京东城区的小吃街
> 能不能介绍一下怎么吃烤鸭？
烤鸭是中国传统菜肴之一，通常用鸭肉、蔬菜、辣椒等调味料制作。您可以选择自己喜欢的配料，然后将鸭肉放入烤盘中，再加上蔬菜和调味料。最后，将烤鸭放入烤箱中烤15-20分钟，享用！
```

**翻译器**
max_memory=350

```
> 你是一个翻译器，请将我说的每一句话翻译成英文。第一句：中国的首都是北京
China’s capital is Beijing.
> 北京也是一个充满活力和创新的城市，拥有许多现代化的建筑和文化活动，如王府井大街、798艺术区等。
Beijing is a vibrant and innovative city with many modern buildings and cultural activities such as Wangfujing Street and 798 Art District.
> 北京的当地小吃有烤鸭、豆汁、炸酱面等等，都是非常地道的美食，可以品尝到当地人的生活气息。
Beijing’s local cuisine includes dishes such as Peking duck, soy sauce noodles, and stir-fried rice cakes, which can give you a taste of the local lifestyle.
```

**扮演**

Since the currently used multi-round dialogue corpus is inserted with a lot of cat lady information, the effect is not good
```
Q：你是谁
A：我是Bot，一款人工智能助手。
Q：接下来我希望你能假装扮演一个高冷的御姐。我们在咖啡厅里遇见了，我会尝试跟你搭讪，但你非常高冷。你会用表情来表达你的高冷，在每轮对话结尾加上表情，如（瞪！）（翻白眼）（冷笑）等。明白了请回复。
A：好的，我会尝试。
Q：好的，这是我的第一句话。美女你好，怎么一个人在咖啡厅里？
A：喵~ 哈哈，我是一位高冷的小姐，今天要去咖啡厅吃咖啡，看看有什么可以帮助我的朋友。
Q：啊我没有需要你帮忙的，我就是看到你一个人，想和你聊聊，可以么
A：喵~ 哈哈，我真的很高冷，所以不太喜欢与人交流。除此之外，我也不太喜欢喝咖啡，因为咖啡会影响我的健康状况。
```

</details>

<details>
<summary><b>知识问答</b></summary>

|                                         |                                                              |                        |
| --------------------------------------- | ------------------------------------------------------------ | ---------------------- |
| 我能用lightning数据线给安卓手机充电吗？ | 是的，你可以使用lightning数据线给安卓手机充电。              | Repetition Penalty=1.3 |
| 为什么天空是蓝色的？                    | 天空是蓝色的，是因为大气层中的氧气和氮气的混合使得光线被散射，形成蓝色的光线。 | Repetition Penalty=2.0 |
| 如何做披萨？                            | 做披萨的步骤如下：<br/>1. 准备披萨饼皮；<br/>2. 准备披萨酱；<br/>3. 准备披萨配料；<br/>4. 将披萨饼皮放在烤盘上；<br/>5. 把披萨酱涂在披萨饼皮上；<br/>6. 把披萨配料涂在披萨饼皮上；<br/>7. 将披萨放入烤箱中烤至金黄色；<br/>8. 取出披萨并享用。 | Repetition Penalty=1.3 |
| 为什么冥王星被踢出太阳系？              | 冥王星被踢出太阳系是因为它的质量比太阳系中其他行星较大，导致它的轨道距离太阳越来越远。 | Repetition Penalty=1.3 |
| 太阳系的主要行星                        | 太阳系的主要行星包括水星、金星、地球、火星、木星、土星、天王星和海王星。 | Repetition Penalty=1.3 |
|                                         |                                                              |                        |
|                                         |                                                              |                        |

</details>

<details>
<summary><b>开放问答</b></summary>

|                                                |                                                              |                        |
| ---------------------------------------------- | ------------------------------------------------------------ | ---------------------- |
| 为什么苹果支付没有在中国流行？                 | 苹果支付在中国没有流行，因为中国的消费者不太愿意使用外国支付应用程序，因为他们更愿意使用中国的支付应用程序，比如支付宝和微信支付等。此外，苹果支付的服务费用比其他支付应用程序更高，这也是一个重要的因素。 | Repetition Penalty=1.0 |
| 我能在咖啡里加盐吗？                           | 是的，你可以在咖啡中加入盐。                                 | Repetition Penalty=1.0 |
| 哪个更受欢迎，TensorFlow还是PyTorch？          | TensorFlow更受欢迎，因为它拥有更广泛的用户群，并且具有更多的支持和资源。 | Repetition Penalty=1.0 |
| 我的宠物走丢了，你能说一些安慰我的话吗？       | 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ 喵~ | Repetition Penalty=1.0 |
| 我的宠物走丢了，你能说一些安慰我的话吗？       | 当你的宠物走出门时，请注意他们的行动，并确保他们不会被遭受外界的威胁。如果有任何问题，请及时联系疫情控制中心或其他相关机构。 | Repetition Penalty=3.0 |
| 我并不是很富裕，我应该花很多钱买最新的手机吗？ | 这取决于您的预算和需求。如果您的预算足够，您可以考虑买最新的手机。但是，您也应该考虑您的需求，比如您需要什么样的功能，以及您是否需要购买具有更多的功能的手机。 | Repetition Penalty=1.0 |
| 如何更好地融入新工作圈子                       | 要更好地融入新工作圈子，可以尝试以下几点：<br/><br/>1. 了解新工作圈子的基本知识和技能；<br/>2. 积极参与工作圈子的活动；<br/>3. 建立良好的工作关系；<br/>4. 寻找领导的帮助；<br/>5. 积极参与团队协作；<br/>6. 学习新技能和知识；<br/>7. 寻找有用的社交网络。 | Repetition Penalty=1.3 |
|                                                |                                                              |                        |
|                                                |                                                              |                        |

</details>

<details>
<summary><b>写信、写文章</b></summary>

|                                                              |                                                              |                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------- |
| 请你帮我写一封签证申请信                                     | 尊敬的招聘团队，<br/><br/>我是一名有着丰富经验的签证申请人，并且对您公司的业务非常感兴趣。我希望能够加入您的团队，为您的公司做出贡献。<br/><br/>我拥有多年的签证申请经验，并且拥有良好的沟通技巧和团队合作能力。我擅长处理复杂的签证申请，并且能够快速解决问题。<br/><br/>我相信我能够在您的公司中发挥重要的作用，并且希望能够成为您的团队成员。<br/><br/>谢谢您的关注，<br/><br/>[你的姓名] | Repetition Penalty=1.3 |
| 我出国需要办理签证，请你帮向领事馆我写一封签证申请信         | 尊敬的领事馆，<br/><br/>我是一名出国旅行者，希望能够在您的领事馆中获得签证，以便进行我的旅行。<br/><br/>我已经准备好了所需要的材料，包括护照、身份证、旅行计划和其他相关证明。我也已经准备好了签证申请表和签证申请文件，以便准备签证。<br/><br/>如果您有任何问题，请随时联系我，我会尽快回复。<br/><br/>谢谢您的帮助！<br/><br/>谢谢，<br/><br/>[您的姓名] | Repetition Penalty=1.3 |
| 请你帮我写一封感谢数学老师的信                               | 尊敬的数学老师，<br/><br/>非常感谢您对我的教育所做出的贡献。您的课程不仅深刻地阐述了数学知识，而且给我提供了有效的实践技巧，使我能够更好地理解和应用数学知识。<br/><br/>您的课程也让我发现数学是一门有趣的学科，并且可以在日常生活中得到应用。您的教学方式令我感到很有趣，同时也让我变得更加有趣。<br/><br/>感谢您对我的教育所做出的贡献！<br/><br/>谢谢！ | Repetition Penalty=2.0 |
| 我得了流感，请帮我写一封假条。                               | 尊敬的领导：<br/><br/>感谢您对我的帮助和支持。我很抱歉地告诉您，我今天感觉不太好，因为我得了流感。我希望能够快速康复，但是我需要一些时间来缓解疼痛。我会尽快完成任务，但是我需要一些时间来缓解疼痛。<br/><br/>感谢您的理解和支持。<br/><br/>谢谢，<br/>[你的名字] | Repetition Penalty=1.3 |
| 你是一个作文生成器，请以《绿水青山就是金山银山》为题写一篇议论文。 | 《绿水青山就是金山银山》是一篇议论文，它探讨了当今社会中的环境问题，以及我们应该如何应对这些问题。这篇议论文指出，当今社会面临着许多环境问题，如气候变化、空气污染、水资源短缺等，这些问题对我们的健康和生活产生了很大的影响。因此，我们需要采取措施来解决这些问题，以保护我们的环境和我们的未来。 | Repetition Penalty=1.0 |
|                                                              |                                                              |                        |
|                                                              |                                                              |                        |

</details>


<details>
<summary><b>写代码</b></summary>

**1、prompt：请用python实现快速排序的代码(Repetition Penalty=1.0)**

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = []
    right = []
    for i in range(0, len(arr)):
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)
```

**2**

- **prompt：你能用python写一下Dijkstra算法的代码吗？**

很抱歉，我不能用Python写Dijkstra算法的代码，因为Dijkstra算法需要使用树结构，而Python不支持树结构。

- **prompt：用python写一下Dijkstra算法的代码**

```python
def dijkstra(graph, source, destination):
    """
    Dijkstra算法的代码
    """
    # 创建一个空字典，用于存储所有节点的距离
    distances = {}
    # 创建一个链表，用于存储所有节点的距离
    path = []
    # 遍历所有节点
    for node in graph:
        # 如果节点是源节点
        if node[0] == source:
            # 记录距离为0
            distances[node[0]] = 0
        # 如果节点是目的地
        elif node[0] == destination:
            # 记录距离为0
            distances[node[0]] = 0
        # 如果节点是递归节点
        else:
            # 遍历所有递归节点
            for neighbor in node[1]:
                # 如果递归节点与源节点之间的距离小于距离
                if distances[neighbor[0]] > distances[node[0]]:
                    # 记录距离
                    distances[node[0]] = distances[neighbor[0]] + 1
```

- **prompt: 用python写一下不带注释的Dijkstra算法的代码**

```python
def dijkstra(graph, source, destination):
    """
    Dijkstra算法
    """
    # 初始化变量
    dist = {}
    visited = {}
    shortest_path = {}
    current_node = {}

    # 遍历所有节点
    for node in graph:
        dist[node] = 0
        visited[node] = False
        current_node[node] = node

    # 遍历所有节点
    for node in graph:
        if not visited[node] and node!= destination:
            dist[node] = max(dist[node], dist[source] + 1)
            visited[node] = True
            shortest_path[node] = source
            current_node[node] = node

    # 返回最短路径
    return shortest_path[destination]
```

**3、Prompt: 请用PyTorch编写一个带ReLU激活函数的全连接层**

```python
python
import torch
from torch.nn.modules.relu import ReLU
from torch.nn.init import initializer

class FullConnected(torch.nn.Module):
    def __init__(self):
        super(FullConnected, self).__init__()
        self.relu = ReLU()
        self.init = initializer.Normal(stddev=0.01)

    def forward(self, x):
        x = self.init(x)
        x = self.relu(x)
        return x
```

**4、Prompt：请写出计算两个数的最大公约数(GCD)的Python代码**

```python
def gcd(a, b):
    if a == 0:
        return b
    else:
        return gcd(b, a % b)
```
</details>


</details>

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

  - 目前我们整合的数据可以在百度网盘或google drive或HuggingFace上下载

    - 链接: https://pan.baidu.com/s/1WSxuhSAotl14ifaAiz5eKw?pwd=b4kb 提取码: b4kb 
    - 链接: https://drive.google.com/file/d/1tzXVhS74m-EtoFot7hEc005LDeZGPit_/view?usp=sharing
    - 链接: https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0

- 上游模型：

  - LLAMA 7B（当然，如果你有更大的机器可以换成13B的，LLAMA13B在数值上优于175B的GPT3）

- lora模型：

  - 我们提供了一个在上面混合数据上训练的lora模型
    - lora model
      -  https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-4000  
      - https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-8000  
      - https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-final 
    - 你也可以从huggingface上加载我们的模型或其他lora模型，加载方式参考[generate.py](https://github.com/Facico/Chinese-Vicuna/blob/master/generate.py)
      - `Facico/Chinese-Vicuna-lora-7b-0.75epoch-belle-and-guanaco`
      - `Facico/Chinese-Vicuna-lora-7b-1.5epoch-belle-and-guanaco`
      - `Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco`
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

**inference并使用gradio生成一个网页（用于指令问答）**

```bash
bash generate.sh
```

- 这里需要注意的参数如下
  - BASE_MODEL，上游模型
  - LORA_PATH，lora模型的checkpoint文件夹
    - 这里要注意的是，lora模型加载的config必须是"adapter_config.json"，模型名字必须是“adapter_model.bin”，不过在训练的时候会自动保存为“pytorch_model.bin”，而"adapter_config.json"和“adapter_model.bin”会在全部训练结束之后保存
      - 如果你是在训练的checkpoint中载入的lora模型，代码里会自动帮你把本地的"config-sample/adapter_config.json"复制到对应目录，并把“pytorch_model.bin”改名为“adapter_model.bin”
    - 也可以是任意的huggingface上对应llama 7B的lora模型，如：`Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco`
  - USE_LOCAL，设置为1时会检查本地模型配置
- 使用的时候，"max_tokens"根据自己电脑的显存来设置，如果生成的内容产生了很多重复信息，可以将"Repetition Penalty"调高

**多轮交互**

由于我们在训练的时候用的基本是指令的prompt，所以闲聊对话能力还比较差，后续将增加这一部分的训练。

```bash
bash interaction.sh
```

- 使用gradio构造的一个简单的交互界面，可以根据自己的机器设置max_memory（它会截取历史对话的后面max_memory部分）

- 这个脚本使用的prompt和generate.sh中使用的不太一样，这个脚本的prompt为对话形式的，如下

  - ```
    The following is a conversation between an AI assistant called Bot and a human user called User.
    ```


同时，为了更好的交互体验，我们自己实现了流式输出（打字机式）交互的chatbot，支持beam search、repetiion penalty的设置，能清空历史记录，选择不同的全局instruction等。

```bash
bash chat.sh
```


## 断点重训/增量训练

考虑到可能程序会中途断开，或者需要在垂直领域的数据上继续训练的情况，我们提供了相应的接口。

下面都是默认多卡脚本，单卡情况请根据上面修改（直接用python运行）

**断点重训**

```bash
finetune_continue.sh
```

- 设置好其中的lora_checkpoint
  - 如果这个目录下有优化器(optimizer.pt)、lr策略(scheduler.pt)等文件，会自动加载并从断掉的地方重新训练
  - 如果这个目录下只有lora相关的模型(adapter_model.bin)和配置(adapter_config.json)，会加载并从头开始训练
- from_data_beginning这个参数表示加载的时候，是否从数据最开始训练（默认否：从数据断开的地方开始训练）

**基于其他数据集继续训练**

当然，你可以选择用上面的脚本，直接从一个已经训练好的lora模型继续训练（不加载任何优化器参数）

你也可以从我们的优化器参数开始继续训练

```bash
finetune_others_continue.sh
```

- from_data_beginning这里会默认从数据最开始训练

这个脚本的逻辑主要是保持学习率一致，如果你的max_steps比我们小，将max_steps和我们训练时的max_steps保持一致，相当于你的数据直接拼在我们断开的数据后面；如果你的数据集比我们大，将直接保持不变



我们目前直接提供1个epoch和2个epoch训练完时的checkpoint

- 1epoch：https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-5800
- 2epoch：https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-11600
- 如果使用我们的checkpoint，你的程序也将从对应的step继续

## **使用纯C++在CPU上进行推理**

详情见`tools`的[readme](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/readme.md)


## **更多工具**

我们还提供了:
- 其他模型权重的下载方式 ( 更快， 8MB/s ) : [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/download_llama.sh)
- 格式转换工具：llama系列模型参数文件的facebook格式 (`consolidated.xx.pth`) 和huggingface格式 (`pytorch_model-000xx-of-000xx.bin`): [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/convert_llama.py)
- LLaMA量化：将模型量化为8bit、4bit、2bit的工具 (`gptq`) : [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/llama_quant.py)

详见[tool readme](https://github.com/Facico/Chinese-Vicuna/tree/master/tools)

## 可能会遇到的问题

在安装并使用本项目时，可能会遇到一些问题，目前遇到的各种问题总结如下：

[Problems](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/problems.md)

# todo

- [x] belle+guanaco(0.72 epoch, 4000 step)
- [x] belle+guanaco(100%)
- [x] 加入更多类似chitchat的对话型语料，增强自由对话的能力
- [x] 增加colab训练+lora载入接口
- [x] 增加了交互能力和打字机式的输出(beam search+流式输出)
- [x] 增加llama的c++推理
- [x] 增加gptq模型量化方法
- [x] 增加增量训练
- [ ] 增加langchain
- [ ] 在多轮instruction数据上训练

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
