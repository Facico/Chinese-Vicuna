This directory offers tools for Vicuna model :
1. to run on CPU (in pure C/C++).
2. quantize the model to 2bit, 4bit, 6bit, 8bit.
---
## Run on CPU (in pure C/C++)
Thanks to the prior work from [Llama.cpp](https://github.com/ggerganov/llama.cpp) and [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)
Notice that:
   - Here are the steps after you have trained a Vicuna lora checkpoint in `lora_path`.
   - The merged model cost 13G disk space for 7B, 37G for 13B, 30B and 65B we haven't test yet due to the limited hardware. Notice that the convertion of model is on cpu and needs large RAM ( peak memory > 64G for 13B, you may need to increase swap size)
   - By default, the 7B,13B,30B,65B checkpoint will be splited into 1,2,4,8 parts during the conversation ( which is fixed in cpp )

1. First you need to merge your lora parameter with original base model and convert them to  `ggml` format for cpp inference.
```
bash prepare_llama_cpp.sh
```
 ( Currently in our code, it will first convert hf model & lora to a merged `consolidated.0x.pth`, where `x` corresponding to num_shards, and convert them to `ggml-model-f16.bin` )
```bash 
python tools/merge_lora_for_cpp.py --lora_path $lora_path
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
Referring to the previous study [pyllama](https://github.com/juncongmoo/pyllama).
you need to install gptq with pip install gptq>=0.0.2 command.
### 1. first, you need to convert the model to huggingface model:
```bash 
python convert_llama.py --ckpt_dir LLaMA_7B --tokenizer_path LLaMA_7B/tokenizer.model --model_size 7B --output_dir LLaMA_7B_hf --to hf
```
### 2. then, quantitative Model:
- Quantize 7B model to 8-bit
```bash
python llama_quant.py decapoda-research/llama-7b-hf c4 --wbits 8 --save pyllama-7B8b.pt --eval
```

- Quantize 7B model to 4-bit with groupsize 128 (the recommended setup 🔥)
```bash
python llama_quant.py decapoda-research/llama-7b-hf c4 --wbits 4 --groupsize 128 --save pyllama-7B4b.pt --eval
```

- Quantize 7B model to 2-bit
```bash
python llama_quant.py decapoda-research/llama-7b-hf c4 --wbits 2 --save pyllama-7B2b.pt --eval
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
- [ ] fix `merge_lora.py` too much space occupation. 
- [ ] fix segmentation fault error due to the fixed `n_ctx` in original code.
- [ ] speedup cpu inference.