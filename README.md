![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/vicuna-llama.png)

# Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model —— 一个中文低资源的llama+lora方案

![GitHub Repo stars](https://img.shields.io/github/stars/Facico/Chinese-Vicuna?style=social) [![HuggingFace badge](https://camo.githubusercontent.com/4a295d6d34ed2c79cfe624ce6358a4be53d4187c883aaa9345fdc322937ce542/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393748756767696e67466163652d4a6f696e2d79656c6c6f77)](https://huggingface.co/Chinese-Vicuna) [![qq join](https://img.shields.io/badge/qq%E7%BE%A4%3A532581765-join-red)](https://jq.qq.com/?_wv=1027&k=47Z6bRjw) [![discord join](https://img.shields.io/badge/discord-join-blue)](https://discord.gg/4FnhmeNHku)
 
 | [English](https://github.com/Facico/Chinese-Vicuna/blob/master/README.md) | [中文](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/readme-zh.md) | [注意事项/NOTEs](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/notes.md)

![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/camel.png)

This is the repo for the Chinese-Vicuna project, which aims to build and share an instruction-following Chinese LLaMA model which can run on a single Nvidia RTX-2080TI, that why we named this project `Vicuna`, small but strong enough ! 

- Why is it called Vicuna：In view of the successful development of alpaca models such as [llama](https://github.com/facebookresearch/llama),[alpaca](https://github.com/tatsu-lab/stanford_alpaca),[guanaco](https://github.com/Guanaco-Model/Guanaco-Model.github.io)，We want to train a Chinese small alpaca like Vicuna.

The repo contains:
- code for finetune the model 
- code for generation based on trained model
- code for run on CPU (fp16 or int4 is support, in purely C++)
- tools to download/convert/quantify original facebook llama.ckpt

This is our instruction demo (with beam-size=4, so you will see 4 process output in the meantime):

https://user-images.githubusercontent.com/72137647/228496412-60043912-f491-430b-848a-599e6edfa5ef.mp4

This is our multi-turn instruction demo (with beam-size=4, so you will see 4 process output in the meantime):

https://user-images.githubusercontent.com/72137647/229739363-1b48f3a9-02a1-46ab-81ee-8c62dc1399b2.mp4




## What‘s New

- March 23, 2023：Released checkpoint-4000 with 50w data training
- March 23, 2023：Deploy the code for fine-tuning and inferencing in colab
- March 23, 2023：Provides code that can be used for inference in pure c++
- March 24, 2023：Released checkpoint-8000 for training about 1.5 epochs on belle+guanaco（100w data）
- March 26, 2023：Provides a quantitative approach
- March 27, 2023：Released  checkpoint-final for training 3 epochs on belle+guanaco
- March 27, 2023：Added multi-round interactive dialog script with alpaca-lora-serve service
- March 28, 2023：Released  our model on [huggingface](https://huggingface.co/Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco)
- March 29, 2023：Added gradio typewriter-like output with beam search, better user interaction support.
- March 29, 2023：Added breakpoint retraining interface to support continued training of other datasets from our checkpoint
- March 29, 2023: Released our new [13B-based lora model](https://huggingface.co/Chinese-Vicuna)
- March 29, 2023: Add more detailed test samples. [performance](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance.md)
- April 1, 2023: Add better support for multi-turn chat in `chat.py` ( Now support 4 generation mode in stream mode/typewriter style: beam search, greedy, sample, beam sample ; We also add cancel button for regeneration )
- April 4, 2023: Add performance for [13B](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-13B.md), which trains on a single 3090.
- April 11, 2023：Released our continuous-finetune on the vertical corpus of Chinese medical quizzes [Chinese-Vicuna-medical](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-medical.md).Provides examples of vertical corpus training

## Possible problems encountered

When installing and using this project, some problems may be encountered, and the various problems encountered so far are summarized as follows:
Before asking questions, take a look at this [Notes](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/notes.md)
## Table of Contents

[Vicuna](https://github.com/Facico/Chinese-Vicuna)

- [what's new](https://github.com/Facico/Chinese-Vicuna#whats-new)
- [what is the meaning](https://github.com/Facico/Chinese-Vicuna#what-is-the-meaning)
- [try on colab](https://github.com/Facico/Chinese-Vicuna#try-on-colab)
- [performance](https://github.com/Facico/Chinese-Vicuna#performance)
  - **Checkpoint-4000**(Facico/Chinese-Vicuna-lora-7b-0.75epoch-belle-and-guanaco)
  - **Checkpoint-8000**(Facico/Chinese-Vicuna-lora-7b-1.5epoch-belle-and-guanaco)
  - **Checkpoint-final**(Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco) and it is used for multiple rounds of dialogue
- [What we need?](https://github.com/Facico/Chinese-Vicuna#what-we-need)
  - code、data、Large Language Model、LORA model、Device
- [How to use](https://github.com/Facico/Chinese-Vicuna#how-to-use)
  - Installing、Multi-gpu training、Single-gpu training、Inference and use gradio to generate a web page(Streaming mode+beam search)、 multi-round interaction and use gradio to generate a web page(Streaming mode+beam search)、Streaming mode base on alpaca-lora-serve
- [inference on CPU with pure C++](https://github.com/Facico/Chinese-Vicuna#inference-on-cpu-with-pure-c)
- [More tools](https://github.com/Facico/Chinese-Vicuna#more-tools)，for more details, see [tool readme](https://github.com/Facico/Chinese-Vicuna/tree/master/tools)
  - ways for faster weight download ( 8MB/s )`download_llama.sh`
  - convert tools between the original facebook checkpoint and huggingface format`convert_llama.py`
  - a quantitative approach that requires less than 4G graphics memory for inference
- [Possible problems encountered](https://github.com/Facico/Chinese-Vicuna#possible-problems-encountered)
- [todo](https://github.com/Facico/Chinese-Vicuna#todo)
- [citation](https://github.com/Facico/Chinese-Vicuna#citation)

## Overview

-  LLaMA paper: https://arxiv.org/abs/2302.13971v1
-  Self-Instruct paper: https://arxiv.org/abs/2212.10560
-  data generation: https://github.com/LianjiaTech/BELLE and https://guanaco-model.github.io/
-  the first work: https://github.com/tatsu-lab/stanford_alpaca

We currently select the combination of BELLE and Guanaco data as our main training dataset. 
We will train on multi-turn instruction data.

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
| [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1Yc38cKifY1_sfTBJpEEKin8m2rxoYXtm?usp=sharing) | Provide a webui for multi-round dialogue interaction         |

## Performance

7B performance can be seen [here](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance.md), 13B is [here](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-13B.md) (which train on a single 3090). There are also people reporting that our method works on 33B and gains good result. 

history-checkpoint performance can be seen [here](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-history.md)


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
<summary><b>checkpoint-final(7B) multi-functional test</b></summary>


<details>
<summary><b>Multi-round interaction,translation and role play</b></summary>

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
<summary><b>Knowledge Q&A</b></summary>

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
<summary><b>Open Q&A</b></summary>

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
<summary><b>Writing letters and articles</b></summary>

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
<summary><b>Coding</b></summary>

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

      

  - The data we currently integrate is available for download on BaiduDownload or Google Drive or HuggingFace

    - link: https://pan.baidu.com/s/1WSxuhSAotl14ifaAiz5eKw?pwd=b4kb   password: b4kb 
    - link: https://drive.google.com/file/d/1tzXVhS74m-EtoFot7hEc005LDeZGPit_/view?usp=sharing
    - link: https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0

- Large Language Model：

  - LLAMA 7B（Of course, if you have a larger machine(such as 3090Ti) can be replaced with a 13B, LLAMA13B is numerically superior to 175B GPT3）

- LORA model：

  - We provide some lora models trained on the above mixed data,
    - lora models 
      - 50w data：https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-4000  
      - 100w data（1.5 epoch）:  https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-8000  
      - all data（3 epoch）:  https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-final
    - You can also load our or other models from huggingface, load it by referring to [generate.py](https://github.com/Facico/Chinese-Vicuna/blob/master/generate.py)
      - `Facico/Chinese-Vicuna-lora-7b-0.75epoch-belle-and-guanaco`
      - `Facico/Chinese-Vicuna-lora-7b-1.5epoch-belle-and-guanaco`
      - `Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco`
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
    - It can also be any lora model on the huggingface corresponding to llama 7B, e.g.: `Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco`
  - USE_LOCAL, which checks the local model configuration when set to 1
- When using, "max_tokens" is set according to your computer's video memory, and if the generated content generates a lot of duplicate information, you can turn up the "Repetition Penalty".



**Multi-round interaction**

As we use the basic command prompt when training, so the ability of small talk conversation is still relatively poor, the follow-up will increase this part of the training.

```bash
bash interaction.sh
```

- A simple interactive interface constructed using gradio, which allows you to set the max_memory according to your machine (it will intercept the max_memory part later in the history conversation)

- The prompt used in this script is not quite the same as the one used in generate.sh. The prompt in this script is in the form of a dialogue, as follows

  - ```
    The following is a conversation between an AI assistant called Bot and a human user called User.
    ```



At the same time, for a better interactive experience, we implemented our own chatbot with streaming output (typewriter-style) interaction, supporting beam search, repetiion penalty settings, the ability to clear history, select different global instruction, etc.

```bash
bash chat.sh
```

## Checkpoint Retraining/Incremental Training

Considering the possibility that the program may be disconnected in the middle of  the process, or the need to continue training on vertical domain data,  we have provided corresponding interfaces. 

 The following are the default multi-GPU scripts. Please modify the single-GPU situation according to the above instruction(run directly in Python) 

**Checkpoint Retraining**

```bash
finetune_continue.sh
```

- Set the `lora_checkpoint`

  - If there are optimizer (optimizer.pt), lr policy  (scheduler.pt), and other files in this directory, they will be automatically loaded and retrained from where they were broken 

  - If there are only LORA related models (adapter_model.bin)  and configurations (adapter_config.json) in this directory, they will be loaded and trained from scratch 

- `from_data_beginning`： The parameter indicates whether to start training from the beginning of the data when loading (default: starting training from the place where the data is disconnected) 

**Incremental Training**

Of course, you can choose to continue training directly from a trained Lora model using the above script (without loading any optimizer parameters)

You can also continue training from our optimizer parameters

```
finetune_others_continue.sh
```

- `from_data_beginning`：This will default to training from the beginning of the data

The logic of this script is mainly to keep the learning rate consistent. If your `max_steps` is smaller than ours, keep `max_steps `consistent with our `max_steps` during training, which is equivalent to putting your data directly behind our disconnected data; if your data set larger than us and will remain directly unchanged.



We currently directly provide checkpoints after 1 epoch and 2 epoch training

- 1epoch：https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-5800
- 2epoch：https://github.com/Facico/Chinese-Vicuna/tree/master/lora-Vicuna/checkpoint-11600
- If you use our checkpoint, your program will also continue from the corresponding step

### Specific cases

- Continue-finetune on the vertical corpus of medicalQA , see here [Chinese-Vicuna-medical](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-medical.md)
## **inference on CPU with pure C++**

Details in `tools` [readme](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/readme.md)

## **More Tools**

We also offer:
- ways for faster weight download ( 8MB/s ) : [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/download_llama.sh)
- convert tools between the original facebook checkpoint (`consolidated.xx.pth`) and huggingface format (`pytorch_model-000xx-of-000xx.bin`): [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/convert_llama.py)
- a quantitative approach that requires less than 4G graphics memory for inference: [link](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/llama_quant.py)

For more details, see [tool readme](https://github.com/Facico/Chinese-Vicuna/tree/master/tools)



# todo

- [x] belle+guanaco(1.5 epoch, 8000 step)
- [x] belle+guanaco(100%)
- [x] Add more chitchat-like conversational corpus to enhance free conversation
- [x] Add colab training + lora loading interface
- [x] Add the interaction capabilities and typewrite-style output(beam search+streaming output)
- [x] Add llama c++ inference
- [x] Add gptq quantification tools
- [x] Add incremental training
- [ ] add langchain
- [ ] train on multi-turn instruction dataset

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
