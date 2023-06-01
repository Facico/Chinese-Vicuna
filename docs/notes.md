# FAQ

仓库里100多个issue了，基本可以涵盖该项目的各种问题，由于很多人都没有找相似issue的习惯，这里整理一份合集。不一定全，但提issue之前请务必看一看有没有和你问题一样的。

关于硬件配置等问题这里也有总结整理，请多加搜索

## 可能遇到的问题：

### 1、GPU版本的问题（非常重要）
由于我们需要使用8bit以及对应的仓库bitsandbytes，8bit优化在适配性会有些问题，**对于GPU显卡的compute capability<7.5的显卡都会有不适配的问题**（这个可以自己去搜一下），这种问题会产生以下问题：

- 1、一个warning：`UserWarning: WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported for your GPU!`，即跑的很慢。我们8bit加载的逻辑，一个是使用GPU 8bit tensor core，一个是在某几个参数会使用fp32 tensor core

- 2、finetune的时候，loss可能会炸。虽然bitsandbytes是适配了，但是会炸（会炸指的是可能loss变得非常大）
  相关的issue
  - https://github.com/Facico/Chinese-Vicuna/issues/39
  - https://github.com/Facico/Chinese-Vicuna/issues/32

- 3、在推理的时候可能会产生乱码。**可以在我们推理脚本中，直接将`device`改成cpu试试能不能产生正常的结果。** 或者使用下面的代码进行测试，假设这个代码叫simple_test.py，使用CUDA_VISIBLE_DEVICES=0 python simple_test.py来运行测试

  - ```python
    import sys
    import torch
    from peft import PeftModel
    import transformers
    from transformers import LlamaTokenizer, LlamaForCausalLM
    
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    BASE_MODEL = "decapoda-research/llama-7b-hf"
    
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True, #改成cpu删除此行
        torch_dtype=torch.float16, #改成cpu删除此行
        device_map="auto",  #{"": "cpu"}
    )
    model.eval()
    inputs = "Hello, Where is the capital of the United States?" #"你好,美国的首都在哪里？"
    input_ids = tokenizer(inputs, return_tensors="pt")['input_ids']
    print(input_ids)
    generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=15,
            )
    print(generation_output)
    print(tokenizer.decode(generation_output[0]))
    
    model = PeftModel.from_pretrained(
            model,
            "./lora-Vicuna/checkpoint-final",
            torch_dtype=torch.float16, #改成cpu删除此行
            device_map={'': 0} #{"": "cpu"}
        )
    
    inputs = "你好,中国的首都在哪里？" #"你好,美国的首都在哪里？"
    input_ids = tokenizer(inputs, return_tensors="pt")['input_ids']
    print(input_ids)                                                                                                      
    generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=15,
            )
    print(generation_output)
    print(tokenizer.decode(generation_output[0]))
    ```

  - 正常输出结果如下，同时可以检查tokenizer（0.13.2）、sentencepiece（sentencepiece  0.1.97）这两个库的版本

  - ```bash
    tensor([[    1, 15043, 29892,  6804,   338,   278,  7483,   310,   278,  3303,
              3900, 29973,    13,  1576,  7483,   310,   278,  3303,  3900,   338,
              7660, 29892,   360, 29889, 29907, 29889,    13]])
     Hello, Where is the capital of the United States?
    The capital of the United States is Washington, D.C.
    
    tensor([[    1, 29871, 30919, 31076, 29892, 30275, 30356, 30210, 31688, 30769,
             30505,   232,   150,   173, 30755, 30882]])
    tensor([[    1, 29871, 30919, 31076, 29892, 30275, 30356, 30210, 31688, 30769,
             30505,   232,   150,   173, 30755, 30882,    13, 30275, 30356, 30210,
             31688, 30769, 30392, 30662, 30675, 30267]])
     你好,中国的首都在哪里？
    中国的首都是北京。
    ```

    相关的issue：

    - https://github.com/Facico/Chinese-Vicuna/issues/2
    - https://github.com/Facico/Chinese-Vicuna/issues/44
    - https://github.com/Facico/Chinese-Vicuna/issues/71
    - https://github.com/Facico/Chinese-Vicuna/issues/121
    - https://github.com/Facico/Chinese-Vicuna/issues/122
### 2、llama模型文件个transformers版本的问题

llama的模型文件有两个下载渠道，他们官方(META AI)的（[agi.gpt4.org](https://agi.gpt4.org/llama/LLaMA/)）和huggingface上的（decapoda_research的）。然而这几个模型上传之后都改过他们的tokenizer（由于看不到官方的模型修改记录所以不知道改成什么样了），对应的tokenizer会和对应的transformers版本进行对齐，transformers中关于llama代码的tokenizers处的逻辑也大改过。

- 这个issue有提供对应最新transformers的llama模型文件：https://github.com/tloen/alpaca-lora/issues/279

现在官方提供的模型可能有点问题（可能要找个对应的transformers版本），我们的建议下载huggingface上的模型文件并固定一个transformers的版本（和我们requirements.txt中的一样）

你可以

```
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@ff20f9cf3615a8638023bc82925573cb9d0f3560
或者
pip install transformers==4.28.1 （4.28.0.dev以上的版本不能用decapoda_research的模型）
```



**llama模型文件的问题**

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/32

- https://github.com/Facico/Chinese-Vicuna/issues/59

**和transformers的tokenizer有关的问题：比如会一直输出不会停止**

- 这是因为不同版本的llama对应的tokenizer中的eos（停止符号）不一样，我们模型中这个是2，可以用下面的代码测试一下

- **！！！eos的id是2不是0，这个很重要，使用前自己测一下！！！**

  使用`python test_tokenizer.py`测试一下

  - 造成这个问题的原因：
    - 老版本的transformers（我们一开始用的4.28.0.dev）加载的是tokenizer.model这个文件，decapoda中这个文件里面eos=2，bos=1.但是它的config里面eos=1，bos=0,因此eos不会错误加载（llama的预训练模型eos=2）
    - 新版的transformers加载的是config，yahma的模型里config是正确的，tokenizer.model也是正确的，因此没有问题。
    - 但是用新版本的transformers加载decapoda就会加载出错误的eos
  **下面看不懂可以忽略，保证eos_token_id=2就好了**

  - 虽然这里有个奇怪的地方就是tokenizer_config.json中写的都是空，我们的tokenizer_config.json如下

  - ```
    {"bos_token": "", "eos_token": "", "model_max_length": 1000000000000000019884624838656, "tokenizer_class": "LLaMATokenizer", "unk_token": ""}
    ```

  - 但我们版本对应的transformers中`tokenizer.eos_token_id`和`tokenizer.bos_token_id`这里，它调用的是sentencepiece的接口求的，这个接口导入的是tokenizer.model这个文件，和tokenizer_config写的不对应（这里在最新版的transformers逻辑中可能改了，但是用我们固定的transformers和）。可以参考这个[issue回复](https://github.com/Facico/Chinese-Vicuna/issues/59#issuecomment-1507135087)

**你也可以把模型改成https://huggingface.co/yahma/llama-7b-hf**
- 这是一个修复了llama的eos的问题的模型

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/55
- https://github.com/Facico/Chinese-Vicuna/issues/59
- https://github.com/Facico/Chinese-Vicuna/issues/71
- https://github.com/Facico/Chinese-Vicuna/issues/140

### 3、peft版本的问题

peft处于一个正在开发的阶段，仓库更新较快，会产生各种各样的新问题，因此我们固定peft的某个commit hash来进行安装，这个在我们的requirements.txt中已经写了，不过有可能你在安装我们的仓库之前已经有相关的环境了所以没有注意。你可以进行下面你的操作：

```bash
pip uninstall peft
pip install git+https://github.com/huggingface/peft@e536616888d51b453ed354a6f1e243fecb02ea08
```

peft版本不同可能会有这个问题，`AttributeError: 'NoneType' object has no attribute 'eval'`：

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/51
- https://github.com/Facico/Chinese-Vicuna/issues/55
- https://github.com/Facico/Chinese-Vicuna/issues/70
- https://github.com/Facico/Chinese-Vicuna/issues/72
- https://github.com/Facico/Chinese-Vicuna/issues/85
- https://github.com/Facico/Chinese-Vicuna/issues/111
- https://github.com/Facico/Chinese-Vicuna/issues/126

### 4、bitsandbytes的问题

一个是版本的问题，建议像我们一样把版本固定到0.37.2，如果版本不一样的话有可能会报错

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/81
- https://github.com/Facico/Chinese-Vicuna/issues/91

一个是bitsandbytes在cuda上加载逻辑的问题，bitsandbytes可能检测不到你的CUDA，然后默认使用cpu之后报错

- 1、检测cuda安装是否正确，使用`echo $CUDA_HOME`看看cuda版本是哪个
- 2、如果正确安装了还是有问题，用对应cuda版本的so去覆盖cpu的so
- 3、windows的话，可以试试cuda11.6：
  - https://github.com/mymusise/ChatGLM-Tuning/issues/42#issuecomment-1502477416
  - https://github.com/Facico/Chinese-Vicuna/issues/64#issuecomment-1521424273

相关的issue：

- https://github.com/TimDettmers/bitsandbytes/issues/156
- https://github.com/Facico/Chinese-Vicuna/issues/2#issuecomment-1482196098
- https://github.com/Facico/Chinese-Vicuna/issues/64

### 5、多卡、单卡以及卡的问题

***多卡能跑单卡报错？***

- **我们的脚本中默认是多卡的配置，多卡的时候用的是torchrun**

但当你只有一张卡的时候使用torchrun可能会报错，这个时候就不要使用torchrun了，而是直接去**指定某一张卡使用python**（使用PEFT不指定GPU会有显存泄露的问题），单卡的时候如我们readme中所示

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py
```

- 需要检查`CUDA_VISIBLE_DEVICES`是否有打错

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/41
- https://github.com/Facico/Chinese-Vicuna/issues/45
- https://github.com/Facico/Chinese-Vicuna/issues/82
- https://github.com/Facico/Chinese-Vicuna/issues/91
- https://github.com/Facico/Chinese-Vicuna/issues/112

***单卡能跑多卡报错？***

- 检查一下指定的对不对，以及自己的卡有没有问题（用nvidia-smi等监视一下GPU）
  - 相关的issue：
  - https://github.com/Facico/Chinese-Vicuna/issues/3

***一些卡自己的问题***

目前有在A4000和A6000在多卡训练遇到超时的问题（运行不正常），可以参考的解决方案：https://www.modb.pro/db/617940

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/49
- https://github.com/Facico/Chinese-Vicuna/issues/53

### 6、网络问题

**gfw的问题**

github的网络问题，中国对github封禁也不是一天两天的事儿了，git clone不下来你可以（具体细节可以询问你喜欢的搜索引擎）

- 1、不走http，走ssh
- 2、终端走代理
- 3、网页下载下来，离线安装
- 4、使用镜像网站

相关的issue（走ssh和镜像的方法相关issue都有）

- https://github.com/Facico/Chinese-Vicuna/issues/50

- https://github.com/Facico/Chinese-Vicuna/issues/80

colab数据无法下载？colab也需要翻墙，可以使用我们提供的百度网盘上的数据

- 相关的issue：https://github.com/Facico/Chinese-Vicuna/issues/67

**gradio的问题**

什么改端口，改IP之类的，这种[gradio仓库](https://github.com/gradio-app/gradio)里写的应该挺详细的

- 相关的issue：https://github.com/Facico/Chinese-Vicuna/issues/61

某些场景不能把link share出去？

- 相关的issue：https://github.com/Facico/Chinese-Vicuna/issues/127



## Q&A

### 有没有交流群？

已经贴在主页了

**QQ群**：532581765

**discord**：https://discord.gg/4FnhmeNHku

有些问题通过邮件交流起来比较繁琐，可以在群里交流。但是由于我们不是主要做这个项目的，还有很多其他事情，问题不会及时回答，这个时候建议里问问群里其他人，或者仔细把项目看一看（有问题直接在项目里搜索一般都能找到）

### 你们项目有啥特点？

llama+中文+**低资源**+垂料训练的方案

我们最初的7B是在3张2080Ti（11G）跑了将近70个小时，用数据并行的方式跑的（1张2080Ti也行，会慢很多）

我们的配置是：lora+8bit+256（长度）（256的时候mirco batch size是4）
- 现在能支持4bit微调+推理（7B的4bit版本大概4-5G左右），2080能微调13B的4bit版本

目前其他大多数项目都是不开8bit+全量微调+2048，然后使用很多A100（由于没有资源，玩不起）

- 初衷就是因为当时大家都是靠比较大的资源来训练的，我们也就是给大家提供了一个低资源的解决方案

- 同时给大家提供低资源的垂料训练方案

### 训练的配置是怎么样的？除了7B能训练吗？大于256截断长度能训练吗？最低配置是什么？

训练的硬件要求主要取决于训练序列长度、`mirco batch size`大小，以及训练时间你是否能接受。

- mirco batch size设置小一点，11G的2080Ti也能跑更大长度的
- mirco batch size设置为1，24G的3090Ti可以训练2048长度的

我们目前总结的大致的训练配置如下(欢迎PR补充)：

> 4090的算力约为3090两倍，A100-40G的int8算力与4090相近

| Model     | GPU    | lora+fp16+512 |      | lora+int8+256 |      | lora+int8+512 |      | lora+int8+2048 |      |
|-----------|--------|---------------|------|---------------|------|---------------|------|----------------|------|
|           |        | speed         | size | speed         | size | speed         | size | speed          | size |
| LLaMA-7B  | 2080Ti |               |      | 0.2h/w        | 11G  |               |      |                |      |
|           | 3090   |               |      |               |      |               |      |                |      |
|           | 4090   | 0.3h/w        | 20G  |               |      | 0.8h/w        |      | 3.5h/w         | 20G  |
| LLaMA-13B | 3090   |               |      | 0.9h/w        |      |               |      | 7.5h/w         | 24G  |
|           | 4090   |               |      |               |      |               |      |                |      |


注意：
- `int8` 仅加载模型显存占用（VRAM）$ \approx $ 硬盘空间大小，比如7B大概8G左右，13B大概14G左右；如果是`fp16`和`fp32`则相应乘2和乘4
- 训练的时候，显存占用和训练的速度和序列长度密切相关，比如序列长度256显存占用不超过11G，这个时候可以在2080Ti上微调7B，序列长度如果是2048，则显存占用会骤增到20G，就要上3090或者4090才能微调7B了；
- 同理，13B在3090/4090上是可以微调的，2048的时候microbatch降到1也是可以跑的
- 另外，有人发现在A100-40G上增大batch没有明显的提速，这可能是因为`int8`比较吃算力（比如相同的配置fp16快于int8），算力吃满后增加batch也不能提高吞吐量，另一方面A100-40G的int8算力其实和4090差不多。

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/65
- https://github.com/Facico/Chinese-Vicuna/issues/84
- https://github.com/Facico/Chinese-Vicuna/issues/94
- https://github.com/Facico/Chinese-Vicuna/issues/107

### 和fastchat他们的vicuna是什么关系？以及效果怎么比？

都是基于llama，然后参照alpaca的方式来进行微调。不过他们是直接使用sharegpt进行微调的，然后配置就是全量以及不开8bit+2048，需要8张A100并开上pytorch自带的fsdp才能跑起来。代码结构并没有什么不同，不过sharegpt比较长，问题质量高一点。我们用到的中文数据集偏短，问题质量差一点。

目前我们的[`chatv1`](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-lora-7b-chatv1)和`VicunaV1.1`有在部分中文问题上能达到接近的表现，见[这里](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-chat.md)。由于我们的`chatv1`在50k条中文指令-对话混合数据上微调3个epoch，可训练参数20w，肯定是不能超越`VicunaV1.1`的全量微调，但不失为一种高效低资源的微调方案。

相关的issue
- https://github.com/Facico/Chinese-Vicuna/issues/48
- https://github.com/Facico/Chinese-Vicuna/issues/89
- https://github.com/Facico/Chinese-Vicuna/issues/108


### llama中文词表比较少？扩充词表有没有必要？

这个issue里面有详细的解答与讨论：https://github.com/Facico/Chinese-Vicuna/issues/12

词表扩充的优势：

- 能编码更多的中文：以为很多token是3对1的，所以中文长度会大打折扣
- 能力问题？
  - fastchat的vicuna其实并没有对词表进行扩充，可以看到他们的效果比做了词表扩充的[这个项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)还强很多。
  - 词表扩充多了是可能会损耗原模型的能力的，除非重新训练一个llama
  - 能够引入一些中文特有的知识
- 难度？词表扩充要进行相应的预训练，也需要比较多的资源，我们也期待的相关工作的进展

### LORA对模型的损耗问题

经过大量实验和以及相关的工作，我们知道lora是可以在**某一领域的微调**中媲美全量微调的。我们知道lora的作用是挖掘基底模型的能力，而吸收新知识的能力较差。

我们现在主要做的是什么呢？挖掘多领域中llama的中文能力。因为llama多领域能力已经很强了，也对中文有初步的理解能力。从fastchat的vicuna可以看出，用少量的中文样本稍微引导一下就有了不错的中文生成能力。从我们提供的7B的结果也能看出，用lora引导中文的生成能力是可行的，在多领域的能力其实还行。

但有个主要差别就是：英文和中文在生成的能力差的还是比较多。尤其是在知识问答上，fastchat的vicuna显著的优于目前很多其他的基于llama的中文模型。

- 目前只能进行简单的猜测，lora针对性引导了llama中中文原本的能力，但在对中英知识对齐这种能力引导的不多。全量微调对中英知识对齐这种能力引导的更多。（由于知识水平有限，只能进行这样的猜测）
  - 造成这样的问题可能的原因：
    - 中英数据比例的问题（我们训练数据中有英文，但是中文占非常大头）
    - 全量微调 VS lora的问题

有待后续探究

### 多个lora进行合并？

可以参考[相关issue](https://github.com/Facico/Chinese-Vicuna/issues/19)

我们后续会做相关的开发。

### RLHF？

可以参考[相关issue](https://github.com/Facico/Chinese-Vicuna/issues/14)

### continue finetune上的问题

#### 垂料数据格式？

可以参考我们医学问答中的[第三种](https://github.com/Facico/Chinese-Vicuna/issues/68)，使用固定的一个instruction，然后让回答更有逻辑。

这里的关键是：

**1、问题定义更加清楚**

我们医学问答其实是一个比较简单的场景，但在一些复杂抽象的场景，你需要把问题定义的尽量清楚，就和你之前催眠chatgpt一样。

可以参考这个issue中[我的回复](https://github.com/Facico/Chinese-Vicuna/issues/115#issuecomment-1524952128)，或者在我们的[performance中的角色扮演](https://github.com/Facico/Chinese-Vicuna/issues/115#issuecomment-1524952128)中的问题定义。

**2、回答设置更有逻辑**

我们医学问答，因为这个数据集本身是一个问题对应多个回答，所以需要去合并，但合并成1/2/3这种形式，其实是有逻辑含义的，在我们给的例子中可以发现他学会的一定的推理。

比如我们角色扮演中的几个任务的格式化处理（小括号，大括号那些）也是会限定逻辑的。把回答设置的更有逻辑类似于思维链，比如你想让他帮你写诗工具人，然后输入给了主题和诗人，你可以把回答格式设置成。

```bash
output："好的，作为一个写诗工具人，我会根据要求的主题“xxxx”，写出尽量符合诗人“xxxx”风格的诗句，如下：xxxx"
```

#### 垂料数据的数量

看这个任务的难度和你想要的效果，比较主流的任务泛化性强，需要的数据少一点。需要引入知识量比较多的任务泛化性差，可能需要的数据多一点。

#### 训练轮数的问题

我们的逻辑是**step**：原始数据是70w，医疗数据是20w，所以从checkpoint-11600(原始的第二个epoch)开始训练，会训练70w个数据对应的step即**5800个step**（5800≈70w/128,因为我们用的是3卡所以其实是70w/120，这个差别不大），即对于新数据会训练70w/20w=3.5个epoch（训练完一个epoch大概需要20w/128=1600个step）

如果你的数据量只有1000，会训练70w/0.1w=700个epoch，训练完一个epoch大概需要1000/128=7.8个step

- 你可以在数据进程中看看经过了多少个epoch（**一般十几二十个epoch就差不多了，可以把save_step设置小一点**）
- **数据显示的epoch量非常大？**
  - 如果从checkpoint-11600开始continue finetune，会把前11600个step按照你现在的数据来计算epoch，比如你的数据是7.8个step一个epoch（1000个数据），可能它会从你开始训练的时候记录的log就是从11600/7.8=1487个epoch开始显示的（**但这个只是显示问题可以忽略，你后面计算epoch的时候可以把这个减到就是你真正的epoch**）

#### 训练了几个epoch是否可以中断？

我们提供的断点重训的代码continue_finetune，可以把相关的checkpoint加载进行继续训练。所以你保存了最近的checkpoint随时可以把程序kill了。

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/38
- https://github.com/Facico/Chinese-Vicuna/issues/52
- https://github.com/Facico/Chinese-Vicuna/issues/68
- https://github.com/Facico/Chinese-Vicuna/issues/93
- https://github.com/Facico/Chinese-Vicuna/issues/95
- https://github.com/Facico/Chinese-Vicuna/issues/103
- https://github.com/Facico/Chinese-Vicuna/issues/106
- https://github.com/Facico/Chinese-Vicuna/issues/115

### 推理效果不行？

#### 推理参数的问题

- 推理参数结果不好可以调整一下repetition penalty，同时beam num需要>1。我们有几个脚本beam num=1的时候没开do sample，可能top k 、top p调整起来差不多。beam num>1时topk topp这些参数没影响。

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/7
- https://github.com/Facico/Chinese-Vicuna/issues/54
- https://github.com/Facico/Chinese-Vicuna/issues/66

- https://github.com/Facico/Chinese-Vicuna/issues/75
- https://github.com/Facico/Chinese-Vicuna/issues/77

**相关参数的含义？**这个去搜一搜比我讲的更清楚。

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/71
- https://github.com/Facico/Chinese-Vicuna/issues/86

#### 不是我们的代码或基底？

可以先用我们的代码试试在提问题

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/62

- https://github.com/Facico/Chinese-Vicuna/issues/71

#### 不兼容8bit的问题

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/2
- https://github.com/Facico/Chinese-Vicuna/issues/44
- https://github.com/Facico/Chinese-Vicuna/issues/71
- https://github.com/Facico/Chinese-Vicuna/issues/121
- https://github.com/Facico/Chinese-Vicuna/issues/122


### missing key！！！

missing key的warning是没有问题的。因为是分两段加载，peft在只加载lora那一段的时候会出现这个问题，但此时的model已经加载过llama参数了。

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/34

- https://github.com/Facico/Chinese-Vicuna/issues/65

### 程序直接结束了没有中间结果保存？？？自己finetune的效果很差？？

我们的batch size是128，相当于每跑128个数据过一个step，然后save_step表示隔多少个step保存一次数据，看看是不是数据量太小，save_step太少了。

- 相关的issue
  - https://github.com/Facico/Chinese-Vicuna/issues/120

在保证你的机子没有8bit的问题后，确保你的训练总step数跟我们差不多

- 相关的issue
  - https://github.com/Facico/Chinese-Vicuna/issues/87
  - https://github.com/Facico/Chinese-Vicuna/issues/92
  - https://github.com/Facico/Chinese-Vicuna/issues/94

### OOM（out of memory）？爆显存

看看我们相关配置，你的显存是否充足，是否有人占着你的显存。

bitsandbytes的版本可能会导致模型保存的时候OOM

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/81
- https://github.com/Facico/Chinese-Vicuna/issues/91

### 找不到config？

在本地产生的checkpoint进行推理的时候把"config-sample/adapter_config.json"复制到你的文件夹中，把pytorch_model.bin改成adapter_model.bin。这个在我们readme中有详细的描述，并在我们的脚本中会自动处理。但如果不用我们的脚本就会遇到这个问题。

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/13

- https://github.com/Facico/Chinese-Vicuna/issues/56
- https://github.com/Facico/Chinese-Vicuna/issues/104
- https://github.com/Facico/Chinese-Vicuna/issues/110

### LLaMATokenizer还是LlamaTokenizer？

这是个warning，是llama他们自己命名的时候有问题，使用对应的tokenizer不会出错，如果使用AutoTokenize会出错

相关的issue：
https://github.com/Facico/Chinese-Vicuna/issues/117



### C++推理的配置

gcc版本

```
>gcc -v
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/lib/gcc/x86_64-linux-gnu/9/lto-wrapper
OFFLOAD_TARGET_NAMES=nvptx-none:hsa
OFFLOAD_TARGET_DEFAULT=1
Target: x86_64-linux-gnu
Configured with: ../src/configure -v --with-pkgversion='Ubuntu 9.4.0-1ubuntu1~20.04.1' --with-bugurl=file:///usr/share/doc/gcc-9/README.Bugs --enable-languages=c,ada,c++,go,brig,d,fortran,objc,obj-c++,gm2 --prefix=/usr --with-gcc-major-version-only --program-suffix=-9 --program-prefix=x86_64-linux-gnu- --enable-shared --enable-linker-build-id --libexecdir=/usr/lib --without-included-gettext --enable-threads=posix --libdir=/usr/lib --enable-nls --enable-clocale=gnu --enable-libstdcxx-debug --enable-libstdcxx-time=yes --with-default-libstdcxx-abi=new --enable-gnu-unique-object --disable-vtable-verify --enable-plugin --enable-default-pie --with-system-zlib --with-target-system-zlib=auto --enable-objc-gc=auto --enable-multiarch --disable-werror --with-arch-32=i686 --with-abi=m64 --with-multilib-list=m32,m64,mx32 --enable-multilib --with-tune=generic --enable-offload-targets=nvptx-none=/build/gcc-9-Av3uEd/gcc-9-9.4.0/debian/tmp-nvptx/usr,hsa --without-cuda-driver --enable-checking=release --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu
Thread model: posix
gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)
```

cmake环境链接可以参考[这里](https://cmake.org/download/)
 由于c++推理部分是我们引入别人的仓库弄进来的，更具体的配置可以参考[alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)

然后记得要编译

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/57
- https://github.com/Facico/Chinese-Vicuna/issues/15

**C++推理效果很差？或者有很多乱码？**

我们更新了GPTQ的量化方案，可以试试，https://github.com/Facico/Chinese-Vicuna/blob/master/tools/readme_zh.md

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/23

- https://github.com/Facico/Chinese-Vicuna/issues/97

### alpaca-serve的问题

我们已经停止对这个工具的维护了，可以使用我们自己的可视化脚本

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/109
- https://github.com/Facico/Chinese-Vicuna/issues/131



### windows+WSL的配置问题

来着@**[robin-human](https://github.com/robin-human)**的提供

1.基础环境配置（显卡驱动、cuda安装、require.txt文件中的工具包安装）
 2.GPU显卡不识别问题处理
如果遇到GPU显卡不识别、文件找不到的问题，请查看这个issue：[TimDettmers/bitsandbytes#52 (comment)](https://github.com/TimDettmers/bitsandbytes/issues/52#issuecomment-1271481182)
 3.两个GPU不同，报错的问题 [pytorch/pytorch#67978 (comment)](https://github.com/pytorch/pytorch/issues/67978#issuecomment-997172378)



和wsl相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/2

- https://github.com/Facico/Chinese-Vicuna/issues/4

- https://github.com/Facico/Chinese-Vicuna/issues/42
- https://github.com/Facico/Chinese-Vicuna/issues/43
- https://github.com/Facico/Chinese-Vicuna/issues/74

## ERROR

### Something went wrong Expecting value: line 1 column 1 (char 0)

报错的指向性不明确，和上面找不到config问题类似

https://github.com/Facico/Chinese-Vicuna/issues/110



### ERROR:torch.distributed.elastic.multiprocessing.api:failed（无前置错误）

如果这个只有程序异常中断都会有这个错误，比如哪个库导致的这个问题，比如程序被kill了之类的,导致这个错误的情况太多了，一般判断哪里有错误不会看这个地方的，只能当做一个程序退出信号。问问题的时候可以看看上面还有没有其他报错。



相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/96

- https://github.com/Facico/Chinese-Vicuna/issues/98
- https://github.com/Facico/Chinese-Vicuna/issues/113



类似问题是中途训练的时候报错的怀疑是硬件问题，暂时还没有解决方案，可以参考的自我诊断方式

- 可以在脚本前面加一个TORCH_DISTRIBUTED_DEBUG=DETAIL，看看有没有更详细的报错信息。
- 将torchrun换成python -m torch.distributed.launch，前面也加上TORCH_DISTRIBUTED_DEBUG=DETAIL看有没有报错（如果没错了把TORCH_DISTRIBUTED_DEBUG=DETAIL删掉）
- 看看单卡会不会报错

### RuntimeError: Trainer requires either a model or model_init argument

transformers版本的问题

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/88

- https://github.com/Facico/Chinese-Vicuna/issues/112

### AttributeError: 'NoneType' object has no attribute 'eval'

peft版本的问题

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/51
- https://github.com/Facico/Chinese-Vicuna/issues/55
- https://github.com/Facico/Chinese-Vicuna/issues/70
- https://github.com/Facico/Chinese-Vicuna/issues/72
- https://github.com/Facico/Chinese-Vicuna/issues/85
- https://github.com/Facico/Chinese-Vicuna/issues/111
- https://github.com/Facico/Chinese-Vicuna/issues/126

### NotImplementedError: Cannot copy out of meta tensor; no data

模型加载的问题，可能你想尝试不使用8bit，但是导致模型的一些参数无效了。

不使用8bit，使用fp16的方法可以使用我们的[finetune_deepspeed.py](https://github.com/Facico/Chinese-Vicuna/blob/master/finetune_deepspeed.py)

相关issue

- https://github.com/Facico/Chinese-Vicuna/issues/83



### RuntimeError: shape '[-1, 32001]' is invalid for input of size **32640000**

**使用的tokenizer和我们不一样**

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/24

**transformers或者llama模型自己的问题**

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/85



### Exception: cublasLt ran into an error!

1、在多卡环境没有指定使用哪张显卡，它会自动在其他显卡上加载（可以用nvidia-smi看看），问题可以见，指定GPU使用CUDA_VISIBLE_DEVICES=xxx
 2、显存不够。比如现在显存上已经跑着一个程序，不够的时候会出现这种情况。
 3、某张卡存在问题。

4、安装torchvision？

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/3

- https://github.com/Facico/Chinese-Vicuna/issues/41



### OutOfMemoryError: CUDA out of memory

如果是在模型保存的时候才报这个错，可能是bitsandbytes版本的问题，要降低一下版本，可以像我们一样固定为0.37.2

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/81
- https://github.com/Facico/Chinese-Vicuna/issues/91

推理时有这个问题的话，可以试着把max_new_token调小一点，beam num调小一点，使用小一点的模型。也可以使用纯cpu推理试试

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/29
- 

### ValueError: Can't find 'adapter_config.json'

相关的issue：

- https://github.com/Facico/Chinese-Vicuna/issues/13

- https://github.com/Facico/Chinese-Vicuna/issues/56
- https://github.com/Facico/Chinese-Vicuna/issues/104
- https://github.com/Facico/Chinese-Vicuna/issues/110

### huggingface_hub.utils._validators.HFValidationError

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/31

### TypeError: dispatch_model() got an unexpected keyword argument 'offload_index'

相关的issue

- https://github.com/Facico/Chinese-Vicuna/issues/18
