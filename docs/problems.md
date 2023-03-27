**1、数据编码问题**

https://github.com/Facico/Chinese-Vicuna/issues/5

我们在处理数据的时候，因为没有强制让json使用非ascii编码，所以非英文部分会自动在json中自动使用ascii转义。

- 由于json的load会自动将这些ascii转义成对应的符号（比如中文），所以并不影响我们的程序运行

下面是一个用bert-base-chinese的一个演示代码，你也可以将其替换成其他的tokenizer

- 使用的数据是sample.json

```python
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
path = "./sample.json"
data = json.load(open(path, "r"))

str_chinese = data[0]['instruction']
print(str_chinese)
print(tokenizer(str_chinese)) #这是tokenizer正常的编码
json.dump(data, open("./sample_test.json", "w")) #这个文件，直接查看的话中文部分都是ascii

data_test = json.load(open("./sample_test.json", "r")) #载入之后是正常的
str_ascii = data_test[0]['instruction']
print(str_ascii)
print(tokenizer(str_ascii))	#由于载入是正常的，不影响tokenizer正常的编码

str_ascii_true = str_chinese.encode('unicode-escape') #我们这里强制转换编码来展示json的转移机制
print('\n')
print(str_ascii_true)
test_json_data = "{" + f"\"instruction\": \"{str_ascii_true}\"" + "}"
test_json_data = test_json_data.replace("\\u", "u")
print(test_json_data) #这个是json将要载入的字典，可以发现里面是上面中文对应的ascii，其中有一个“b”字符请忽略，这是bytes对象的字符串
ascii_test_data = json.loads(test_json_data)
print(ascii_test_data) #用json载入之后发现自动转义成正常的中文

json.dump(data, open("./sample_test_utf8.json", "w"), ensure_ascii=False, indent=2)
#如果想json能查看的舒服一点，可以增加后面这两个参数，让其自动缩进同时不编码ascii
```

正常输出如下：

```bash
用一句话描述地球为什么是独一无二的。\n

{'input_ids': [101, 4500, 671, 1368, 6413, 2989, 6835, 1765, 4413, 711, 784, 720, 3221, 4324, 671, 3187, 753, 4638, 511, 139, 156, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
用一句话描述地球为什么是独一无二的。\n

{'input_ids': [101, 4500, 671, 1368, 6413, 2989, 6835, 1765, 4413, 711, 784, 720, 3221, 4324, 671, 3187, 753, 4638, 511, 139, 156, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


b'\\u7528\\u4e00\\u53e5\\u8bdd\\u63cf\\u8ff0\\u5730\\u7403\\u4e3a\\u4ec0\\u4e48\\u662f\\u72ec\\u4e00\\u65e0\\u4e8c\\u7684\\u3002\\\\n\\n'
{"instruction": "b'\u7528\u4e00\u53e5\u8bdd\u63cf\u8ff0\u5730\u7403\u4e3a\u4ec0\u4e48\u662f\u72ec\u4e00\u65e0\u4e8c\u7684\u3002\\\\n\\n'"}
{'instruction': "b'用一句话描述地球为什么是独一无二的。\\\\n\\n'"}
```

如果希望能清晰地查看里面的内容，可以使用上面代码的最后一行。先将数据load后

- 使用ensure_ascii=False让其不自动转换成ascii
- 使用indent调整json的缩进



**2、环境问题**

由于不同机器不同版本会有各种不同的问题。

- ddp跑单卡的问题

  - https://github.com/Facico/Chinese-Vicuna/issues/4
  - 由于finetune.sh使用的是torchrun来运行的，可能存在torch和python版本之间的不对应问题。因为单卡不需要ddp，此时可以直接使用python来运行。参照readme中的单卡指令。

- ddp跑多卡

  - https://github.com/Facico/Chinese-Vicuna/issues/3

  - 首先注意是不是自己机器的问题，然后再注意版本的问题，一下是一个python3.10能用的配置参考

  - ```
    torch                    1.13.1
    torchtyping              0.1.4
    torchvision              0.14.1
    absl-py                  1.4.0
    accelerate               0.15.0
    aiodns                   3.0.0
    aiofiles                 23.1.0
    aiohttp                  3.8.3
    aiosignal                1.3.1
    altair                   4.2.2
    anyio                    3.6.2
    appdirs                  1.4.4
    async-timeout            4.0.2
    attrs                    22.2.0
    beautifulsoup4           4.11.2
    bitsandbytes             0.37.0
    Brotli                   1.0.9
    cachetools               5.3.0
    certifi                  2022.12.7
    cffi                     1.15.1
    charset-normalizer       2.1.1
    click                    8.1.3
    contourpy                1.0.7
    cpm-kernels              1.0.11
    cycler                   0.11.0
    datasets                 2.8.0
    deepspeed                0.7.7
    dill                     0.3.6
    distlib                  0.3.6
    docker-pycreds           0.4.0
    einops                   0.6.0
    entrypoints              0.4
    evaluate                 0.4.0
    fastapi                  0.95.0
    ffmpy                    0.3.0
    filelock                 3.9.0
    fire                     0.5.0
    flash-attn               0.2.8
    fonttools                4.39.2
    frozenlist               1.3.3
    fsspec                   2023.3.0
    gdown                    4.6.4
    gensim                   3.8.2
    gitdb                    4.0.10
    GitPython                3.1.31
    google-auth              2.16.2
    google-auth-oauthlib     0.4.6
    gradio                   3.23.0
    grpcio                   1.51.3
    h11                      0.14.0
    hjson                    3.1.0
    httpcore                 0.16.3
    httpx                    0.23.3
    huggingface-hub          0.13.3
    icetk                    0.0.5
    idna                     3.4
    inflate64                0.3.1
    Jinja2                   3.1.2
    joblib                   1.2.0
    jsonlines                3.1.0
    jsonschema               4.17.3
    kiwisolver               1.4.4
    linkify-it-py            2.0.0
    loguru                   0.6.0
    loralib                  0.1.1
    Markdown                 3.4.1
    markdown-it-py           2.2.0
    MarkupSafe               2.1.2
    matplotlib               3.7.1
    mdit-py-plugins          0.3.3
    mdurl                    0.1.2
    msgpack                  1.0.4
    multidict                6.0.4
    multiprocess             0.70.14
    multivolumefile          0.2.3
    networkx                 3.0
    ninja                    1.11.1
    nltk                     3.8.1
    numpy                    1.24.2
    nvidia-cublas-cu11       11.10.3.66
    nvidia-cuda-nvrtc-cu11   11.7.99
    nvidia-cuda-runtime-cu11 11.7.99
    nvidia-cudnn-cu11        8.5.0.96
    nvidia-ml-py             11.525.84
    nvitop                   1.0.0
    oauthlib                 3.2.2
    openai                   0.27.2
    orjson                   3.8.8
    packaging                23.0
    pandas                   1.5.3
    pathtools                0.1.2
    peft                     0.3.0.dev0
    Pillow                   9.4.0
    pip                      22.3.1
    platformdirs             3.1.0
    protobuf                 3.20.1
    psutil                   5.9.4
    py-cpuinfo               9.0.0
    py7zr                    0.20.4
    pyarrow                  11.0.0
    pyasn1                   0.4.8
    pyasn1-modules           0.2.8
    pybcj                    1.0.1
    pycares                  4.3.0
    pycparser                2.21
    pycryptodomex            3.17
    pydantic                 1.10.4
    pydub                    0.25.1
    Pygments                 2.14.0
    pyparsing                3.0.9
    pyppmd                   1.0.0
    pyrsistent               0.19.3
    PySocks                  1.7.1
    python-dateutil          2.8.2
    python-multipart         0.0.6
    pytz                     2022.7.1
    PyYAML                   6.0
    pyzstd                   0.15.4
    ray                      2.3.0
    regex                    2022.10.31
    requests                 2.28.2
    requests-oauthlib        1.3.1
    responses                0.18.0
    rfc3986                  1.5.0
    rich                     13.3.2
    rouge-score              0.1.2
    rsa                      4.9
    scikit-learn             1.2.0
    scipy                    1.10.1
    semantic-version         2.10.0
    sentencepiece            0.1.97
    sentry-sdk               1.16.0
    setproctitle             1.3.2
    setuptools               65.6.3
    six                      1.16.0
    smart-open               6.3.0
    smmap                    5.0.0
    sniffio                  1.3.0
    soupsieve                2.4
    starlette                0.26.1
    tabulate                 0.9.0
    tensorboard              2.12.0
    tensorboard-data-server  0.7.0
    tensorboard-plugin-wit   1.8.1
    termcolor                2.2.0
    texttable                1.6.7
    threadpoolctl            3.1.0
    tokenizers               0.13.2
    toolz                    0.12.0
    torch                    1.13.1
    torchtyping              0.1.4
    torchvision              0.14.1
    tqdm                     4.65.0
    transformers             4.28.0.dev0
    trlx                     0.3.0
    typeguard                2.13.3
    typing_extensions        4.5.0
    uc-micro-py              1.0.1
    urllib3                  1.26.14
    uvicorn                  0.21.1
    virtualenv               20.20.0
    wandb                    0.13.10
    websockets               10.4
    Werkzeug                 2.2.3
    wheel                    0.38.4
    xxhash                   3.2.0
    yarl                     1.8.2
    ```



**3、输出乱码问题**

- https://github.com/Facico/Chinese-Vicuna/issues/2
- 可以使用下面的测试代码，检验一下英文输入，英文输出，中文输入，中文输出等是否会有问题。如果有问题可能是transformers、tokenizers、sentencepiece等依赖版本的问题（参考上面配置），如果在终端输出解决了这个问题webui上应该是不会有问题的。

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
      load_in_8bit=True,
      torch_dtype=torch.float16,
      device_map="auto",
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
          "./lora-Vicuna/checkpoint-4000",
          torch_dtype=torch.float16,
          device_map={'': 0}
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

- 因为这个是一个比较简单的例子，生成的时候没有加参数控制，在webui那里会有参数控制的，比如Repetition Penalty等