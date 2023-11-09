|[English](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/readme.md)|[中文](https://github.com/Facico/Chinese-Vicuna/blob/master/tools/readme_zh.md)|

This directory offers tools for Vicuna model :
1. to run on CPU (in pure C/C++).
2. quantize the model to 2bit, 4bit, 6bit, 8bit.
---
## Run on CPU (in pure C/C++)
Thanks to the prior work from [Llama.cpp](https://github.com/ggerganov/llama.cpp) and [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)
Notice that:
   - Here are the steps after you have trained a Vicuna lora checkpoint in `lora_path`.
   - The merged model cost 13G disk space for 7B, 37G for 13B, 30B and 65B we haven't test yet due to the limited hardware. Notice that the convertion of model is on cpu and needs large RAM ( peak memory > 64G for 13B, you may need to increase swap size)
   - By default, the 7B,13B,30B,65B checkpoint will be splitted into 1,2,4,8 parts during the conversation ( which is fixed in cpp )

1. First you need to merge your lora parameter with original base model and convert them to  `ggml` format for cpp inference.
```
bash prepare_llama_cpp.sh
```
 ( Currently in our code, it will first convert hf model & lora to a merged `consolidated.0x.pth`, where `x` corresponding to num_shards, and convert them to `ggml-model-f16.bin` )
```bash 
python tools/merge_lora.py --lora_path $lora_path
```

1. next, go to the `vicuna.cpp` directory and start to chat pure in CPU & C++ !
```bash
cd tools/vicuna.cpp
make chat 
# we also offer a Makefile.ref, which you can call it with `make -f Makefile.ref `
./chat -m $ggml-path

```
[Optional] you can convert above ggml to int4 (`ggml-model-q4_0.bin`) and use it in chat,  (but the performance is worse)
```bash
make quantize
./quantize.sh
```

---
## Quantize LLaMA
Provides a quantitative approach that allows you to use the LLaMA-7B model for inference on devices with less than 4G graphics memory.
Referring to the previous study [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa).
The version of transformers used for local testing is 4.29.0.dev0.
### 1. first, you need to convert the model to huggingface model:
```bash 
python convert_llama.py --input_dir /model/llama-7b --model_size 7B --output_dir ./llama-hf
```
### 2. then, quantitative Model:
- Quantize 7B model to 8-bit
```bash
CUDA_VISIBLE_DEVICES=0 python quant_llama.py ./llama-hf/llama-7b wikitext2 --wbits 8 --true-sequential --act-order --groupsize 128 --save llama7b-8bit-128g.pt
```

- Quantize 7B model to 4-bit with groupsize 128 (recommend)
```bash
CUDA_VISIBLE_DEVICES=0 python quant_llama.py ./llama-hf/llama-7b wikitext2 --wbits 4 --true-sequential --act-order --groupsize 128 --save llama7b-4bit-128g.pt
```

- Quantize 7B model to 2-bit
```bash
CUDA_VISIBLE_DEVICES=0 python quant_llama.py ./llama-hf/llama-7b wikitext2 --wbits 2 --true-sequential --act-order --groupsize 128 --save llama7b-2bit-128g.pt
```
### 3. Generate results directly or use gradio on the web:
- Reasoning from the input text
```bash
python quant_generate.py --model_path ./llama-hf/llama-7b --quant_path llama7b-4bit-128g.pt --wbits 4 --groupsize 128 --text "the mean of life is"
```
- use gradio to generate a web page:
```bash
python quant_generate.py --model_path ./llama-hf/llama-7b --quant_path llama7b-4bit-128g.pt --wbits 4 --groupsize 128 --gradio
```

#### LLaMA-7B Generate results and graphics memory usage：
- 8bit [8.5G MEM] [Download](https://huggingface.co/Chinese-Vicuna/llama7b_8bit_128g).
```text
the mean of life is 70 years.
the median age at death in a population, regardless if it's male or female?
```
- 4bit [5.4G MEM] [Download](https://huggingface.co/Chinese-Vicuna/llama7b_4bit_128g).
```text
the mean of life is 70 years.
the median age at death in africa was about what?
```
- 2bit [4G MEM] [Download](https://huggingface.co/Chinese-Vicuna/llama7b_2bit_128g).
```text
the mean of life is a good., and it’s not to be worth in your own homework for an individual who traveling on my back with me our localities that you can do some work at this point as well known by us online gaming sites are more than 10 years old when i was going out there around here we had been written about his time were over all sited down after being spent from most days while reading between two weeks since I would have gone before its age site;...
```
---
## Quantize Models with AWQ
Provides AWQ method for easy 4-bit level quantization.
Please install autoAWQ with
```bash
pip install autoawq
```
- Quantize model to AWQ 4-bit.
```bash
python awq/basic_quant.py --model_path models/llama-7b-hf --save llama-7b-hf-awq
```

- Inference with quantized models.
```bash
python awq/basic_generate.py --quant_path llama-7b-hf-awq
```

- Run benchmarks of speed and memory usage of processing context (prefill) and generating tokens (decoding).
```bash
python awq/benchmark.py --model_path llama-7b-hf-awq
```

- Evaluate perplexity of quantized model or unquantized model
```bash
python awq/eval.py --model_path llama-7b-hf-awq
python awq/eval.py --use_pretrained --model_path models/llama-7b-hf
```
---

TODO:
- [ ] fix `merge_lora.py` too much space occupation. 
- [ ] fix segmentation fault error due to the fixed `n_ctx` in original code.
- [ ] speedup cpu inference.
