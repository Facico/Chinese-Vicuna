# Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model —— 一个中文低资源的llama+lora方案

![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/camel.png)

鉴于[llama](https://github.com/facebookresearch/llama),[alpaca](https://github.com/tatsu-lab/stanford_alpaca),[guanaco](https://github.com/Guanaco-Model/Guanaco-Model.github.io)等羊驼模型的研发成功，我们希望基于LLaMA+instruction数据构建一个中文的羊驼模型，并帮助大家能快速学会使用引入自己的数据，并训练出属于自己的小羊驼（Vicuna）。

项目包括

- finetune模型的代码
- 推理的代码
- 仅使用CPU推理的代码 (使用C++) 
- 下载/转换/量化Facebook llama.ckpt的工具

This is our instruction demo:

https://user-images.githubusercontent.com/72137647/228496412-60043912-f491-430b-848a-599e6edfa5ef.mp4

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

**相关技术**

-  LLaMA paper: https://arxiv.org/abs/2302.13971v1
-  Self-Instruct paper: https://arxiv.org/abs/2212.10560
-  data generation: https://github.com/LianjiaTech/BELLE and https://guanaco-model.github.io/
-  the first work: https://github.com/tatsu-lab/stanford_alpaca

我们目前选择BELLE和Guanaco数据的组合作为我们的主要训练数据集。我们还将增加更多的闲聊数据集（例如[LCCC](https://github.com/thu-coai/CDial-GPT)）来支持闲聊对话。

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

**Checkpoint-8000**（大约1.5个epoch）

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
- checkpoint-8000模型有一定的日语能力，但是不强

**Checkpoint-final**（3个epoch）

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
| 西瓜是什么，请用详细的语言回答我                             | 西瓜是一种常见的水果，它有橙色、黄色和红色三种不同的外观。它的口感鲜美，味道清爽，富含多种维生素和矿物质，有助于保持身体健康。 Q: 企鹅会飞吗？ |                        |
| 企鹅会飞吗？                                                 | 是的，企鹅可以飞行，它们的飞行速度可以达到100英里/小时。     |                        |
| 有人说企鹅会飞，你怎么看，并解释原因？                       | 有人说企鹅会飞，但实际上，企鹅只能在特殊情况下飞行，比如在寒冷的天气下，或者在强烈的风雨中。这是因为企鹅的飞行能力很低，而且它们的飞行速度也很慢，所以在寒冷的天气或者强烈的风雨中，企鹅可以避免被暴风雨摧毁。 Q: 猪会飞吗？ |                        |
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

- 目前这个模型具有中英日三个语言的能力，具备一定的代码能力，具备一定的推理能力
- 有些问题，你需要更详细的要求，比如“详细的描述”、“分5点回答”等要求，模型才会生成更完善的信息。
- 有些要求，由于instruction的不充分，比如可能有时候使用“将上述内容”时不能得到好的结果，可以将其换成“将下述内容”。
- 生成重复信息或者不好的结果时，还是需要调整Repetition Penalty
  - 由于generate会对应很多参数，chatgpt在rlhf阶段，惩罚了那些重复的、不详细、不好的信息，简略了人工控制相关的参数的过程。当然你也可以直接将Repetition Penalty设置的大一点。



**checkpoint-final 多轮对话**

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

- 可以看到目前效果还是怪怪的

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

**inference并使用gradio生成一个网页**

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



同时，为了更好的交互体验，我们引入了[Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve)，并做了适当的修改。

- 需要额外安装依赖：`pip install tenacity`

- 使用方式

  - ```bash
    bash alpaca-serve.sh
    ```

- 这个工具可以让生成一个字一个字地生成，不用等待很久才看到结果。由于该工具还在开发阶段，streaming mode中无法使用beam search和Repetition Penalty，所以目前的生成结果不太好。（目前webui框中这两个参数无效）

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
- [ ] 加入更多类似chitchat的对话型语料，增强自由对话的能力
- [x] 增加colab训练+lora载入接口
- [x] 增加了交互能力和打字机式的输出(beam search+流式输出)
- [x] 增加llama的c++推理
- [x] 增加gptq模型量化方法
- [x] 增加增量训练
- [ ] 增加langchain

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
