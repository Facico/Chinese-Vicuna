本目录主要提供Vicuna model相关的工具:
1. 使用纯C++推理
2. 使用GPTQ量化到2bit, 4bit, 6bit, 8bit.
---
## 使用纯C++推理
感谢之前的工作： [Llama.cpp](https://github.com/ggerganov/llama.cpp) 、 [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp), 请注意

   - 这里的步骤应该在你训练完了lora再进行.
   - 合并后的checkpoint对于7B模型大概消耗13G磁盘空间，对于13B模型大概消耗37G, 30B和65B由于我们有限的设备条件没有测试. 注意在转换过程中会消耗很大的内存 ( 比如13B可能超过64G，但你可以通过提高swap空间解决 )
   - 另外， 7B,13B,30B,65B的checkpoint分别默认被分成1,2,4,8片 ( 这也是cpp里边固定的设置 )

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
- [ ] fix `merge_lora.py` too much space occupation. 
- [ ] fix segmentation fault error due to the fixed `n_ctx` in original code.
- [ ] speedup cpu inference.