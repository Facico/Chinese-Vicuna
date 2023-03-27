![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/vicuna-llama.png)

# Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model —— 一个中文低资源的llama+lora方案

 | [English](https://github.com/Facico/Chinese-Vicuna/blob/master/README.md) | [中文](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md) |

![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/camel.png)

This is the repo for the Chinese-Vicuna project, which aims to build and share an instruction-following Chinese LLaMA model which can run on a single Nvidia RTX-2080TI, that why we named this project `Vicuna`, small but strong enough ! 

- Why is it called Vicuna：In view of the successful development of alpaca models such as [llama](https://github.com/facebookresearch/llama),[alpaca](https://github.com/tatsu-lab/stanford_alpaca),[guanaco](https://github.com/Guanaco-Model/Guanaco-Model.github.io)，We want to train a Chinese small alpaca like Vicuna.

The repo contains:
- code for finetune the model 
- code for generation based on trained model
- code for run on CPU (fp16 or int4 is support, in purely C++)
- tools to download/convert/quantify original facebook llama.ckpt

## What‘s New

- March 23, 2023：Released checkpoint-4000 with 50w data training
- March 23, 2023：Deploy the code for fine-tuning and inferencing in colab
- March 23, 2023：Provides code that can be used for inference in pure c++
- March 24, 2023：Released checkpoint-8000 for training about 1.5 epochs on belle+guanaco（100w data）
- March 26, 2023：Provides a quantitative approach
- March 27, 2023：Released  checkpoint-final for training 3 epochs on belle+guanaco
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

- This LORA model is the result of 30 hours of training with a 2080Ti (checkpoint-4000), about 50w data

**Checkpoint-8000**(about 1.5 epoch on belle+guanaco)

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

**Checkpoint-final** (3 epoch on belle+guanaco)

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

- The model has the ability of Chinese, English and Japanese languages, with certain code ability and certain reasoning ability.
- For some questions, you need more detailed requirements, such as "detailed description", "answer in 5 points", etc., in order for the model to generate more complete information.
- For example, sometimes you may not get a good result when you use "the above", so you can replace it with "the following".
- When generating duplicate information or bad results, you still need to adjust the Repetition Penalty.
  - Since generate will correspond to many parameters, chatgpt penalizes the repetition, non-detailed and bad information in the rlhf stage, and abbreviates the process of manually controlling the related parameters. Of course you can also directly set the Repetition Penalty to a larger size.

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

  - We provide some lora models trained on the above mixed data,
    - lora models 
      - 50w data：https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-4000  
      - 100w data（1.5 epoch）:  https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-8000  
      - all data（3 epoch）:  https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-final
    - Since the model is relatively small, it is temporarily uploaded on github(about 30M).More lora models will be uploaded later on huggingface or on the BaiduDownload or Google Drive
    - The model uses 8bit+lora+256 tokens

- Device：

  - Training：A 2080Ti is sufficient. Since the data length is within 256, it takes about 9G of video memory.
    - 70w of data, 3 epochs, a 2080Ti about 200h
  - Inference：A 2080Ti is all you need。
  - CPU Inference is also support! please go to see [`tools`](https://github.com/Facico/Chinese-Vicuna/blob/master/tools)

## How to use

**Installation**

```
git clone https://github.com/Facico/Chinese-Vicuna
pip install -r requirements.txt
```

Local python environment is 3.8, torch is 1.13.1, CUDA is 12

NOTE: python3.11 has a known `torchrun` bug, details [here](https://github.com/facebookresearch/llama/issues/86)

**Multi-gpu Training**

```bash
bash finetune.sh
```

- The parameters to note here are as follows
  - TOT_CUDA, fill in the GPU number to be used, such as `TOT_CUDA="0,1,2,3"`
  - PORT, fill in the corresponding port
  - DATA_PATH，fill in the corresponding data location in the format of json
  - OUTPUT_PATH，fill in the relative path to save the model
  - MODEL_PATH，path of LLM
  - wandb：This is a training visualization tool that is not turned on by default in the script, and can be turned on by adding "--wandb" to the script

**Single-gpu Training**

```
python finetune.py --data_path merge.json --test_size 2000
```

- The test_size cannot be larger than the data size

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

## **inference on CPU with pure C++**

Details in `tools` [readme](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/readme.md)

## **More Tools**

We also offer:
- ways for faster weight download ( 8MB/s ) : [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/download_llama.sh)
- convert tools between the original facebook checkpoint (`consolidated.xx.pth`) and huggingface format (`pytorch_model-000xx-of-000xx.bin`): [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/convert_llama.py)
- a quantitative approach that requires less than 4G graphics memory for inference: [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/llama_quant.py)

## Possible problems encountered

When installing and using this project, some problems may be encountered, and the various problems encountered so far are summarized as follows:

[Problems](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/problems.md)

# todo

- [x] belle+guanaco(1.5 epoch, 8000 step)
- [x] belle+guanaco(100%)
- [ ] Add more chitchat-like conversational corpus to enhance free conversation
- [x] Add colab training + lora loading interface
- [x] Add the interaction capabilities
- [x] Add llama c++ inference
- [x] Add gptq quantification tools

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
