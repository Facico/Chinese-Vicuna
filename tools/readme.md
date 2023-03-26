This directory offers tools for Vicuna model :
1. to run on CPU (in pure C/C++).
2. quantize the model to 2bit, 4bit, 6bit, 8bit.
---
## Run on CPU (in pure C/C++)
Thanks to the prior work from [Llama.cpp](https://github.com/ggerganov/llama.cpp) and [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)

Here are the steps if you have trained a Vicuna lora checkpoint in `lora_path` ( When using 7B model, it will get 13G merge model checkpoint and ggml checkpoint so we don't upload them.):

1. first you need to merge your lora parameter with original base model (e.g. `decapoda-research/llama-7b-hf`) and save a merged `pytorch_model.bin` from root directory. normally you will get `consolidated.00.pth` and `params.json` in output dir
```bash 
python tools/merge_lora_for_cpp.py --lora_path $lora_path
```
2. then, convert above saved `pytorch_model.bin` to ggml format, by default get `ggml-model-f16.bin` in the same dir as `pytorch_model.bin`
```bash
python tools/convert_pth_to_ggml.py 
```
3. next, go to the `vicuna.cpp` directory and start to chat pure in CPU & C++ !
```bash
cd tools/vicuna.cpp
make chat
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
Referring to the previous study [pyllama](https://github.com/juncongmoo/pyllama).
you need to install gptq with pip install gptq>=0.0.2 command.
### 1. first you need to convert model as huggingface model  by:
```bash 
python --ckpt_dir LLaMA_7B --tokenizer_path LLaMA_7B/tokenizer.model --output_dir LLaMA_7B_hf --to hf
```
### 2. then, quantitative Model:
- Quantize 7B model to 8-bit
```bash
python llama_quant.py decapoda-research/llama-7b-hf c4 --wbits 8 --save pyllama-7B8b.pt
```

- Quantize 7B model to 4-bit with groupsize 128 (the recommended setup 🔥)
```bash
python llama_quant.py decapoda-research/llama-7b-hf c4 --wbits 4 --groupsize 128 --save pyllama-7B4b.pt
```

- Quantize 7B model to 2-bit
```bash
python llama_quant.py decapoda-research/llama-7b-hf c4 --wbits 2 --save pyllama-7B2b.pt
```
### 3. finally, inference and use gradio to generate a web page:
```bash
CUDA_VISIBLE_DEVICES=0
python generate_quant.py \
    --model_path "decapoda-research/llama-7b-hf" \
    --quant_path "pyllama-7B2b.pt" \
    --wbits 2
```

The inference with 7B 2bit model requires only 3.8GB GPU memory when beam search is set to  1.

---
TODO:
- [ ] speedup cpu inference.
- [ ] fix segmentation fault error.