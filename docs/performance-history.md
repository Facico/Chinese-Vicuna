模型迭代中的测试样例



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