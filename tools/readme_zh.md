本目录主要提供Vicuna model相关的工具:
1. 使用纯C++推理
2. 使用GPTQ量化到2bit, 4bit, 6bit, 8bit.
---
## 使用纯C++推理
感谢之前的工作： [Llama.cpp](https://github.com/ggerganov/llama.cpp) 、 [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp), 请注意

   - 这里的步骤应该在你训练完了lora再进行.
   - 合并后的checkpoint对于7B模型大概消耗13G磁盘空间，对于13B模型大概消耗37G, 30B和65B由于我们有限的设备条件没有测试. 注意在转换过程中会消耗很大的内存 ( 比如13B可能超过64G，但你可以通过提高swap空间解决 )
   - 另外， 7B,13B,30B,65B的checkpoint分别默认被分成1,2,4,8片 ( 这也是cpp里边固定的设置 )

1.首先，你需要将你的lora参数与原始模型合并，并将它们转换为`ggml`格式，用于cpp推理。
```
bash prepare_llama_cpp.sh
```
 ( 在我们的代码中，首先将hf模型和lora转换为合并的`consolidated.0x.pth`，其中`x`对应num_shards，并将它们转换为`ggml-model-f16.bin`。 )
```bash 
python tools/merge_lora_for_cpp.py --lora_path $lora_path
```

1. 接下来，进入`vicuna.cpp`目录，开始使用CPU和C++进行聊天 !
```bash
cd tools/vicuna.cpp
make chat 
# we also offer a Makefile.ref, which you can call it with `make -f Makefile.ref `
./chat -m $ggml-path

```
[Optional]你可以将上述ggml转换为int4（`ggml-model-q4_0.bin`）然后进行聊天（但性能会有所损失）。
```bash
make quantize
./quantize.sh
```

---
## Quantize LLaMA
提供了一种定量的方法，允许你在图形内存小于4G的设备上使用LLaMA-7B模型进行推理，参考之前的研究[pyllama]（https://github.com/juncongmoo/pyllama）。
运行下面的代码前，你需要用 pip install gptq>=0.0.2 命令来安装 gptq。
### 1. 首先需要确保模型为huggingface格式。如果不是，可以通过下面的命令转换:
```bash 
python --ckpt_dir LLaMA_7B --tokenizer_path LLaMA_7B/tokenizer.model --output_dir LLaMA_7B_hf --to hf
```
### 2. 然后进行模型量化，下面分别是量化为8bit、4bit、2bit的方法:
- 将LLaMA-7B的模型量化为8-bit
```bash
python llama_quant.py decapoda-research/llama-7b-hf c4 --wbits 8 --save pyllama-7B8b.pt
```

- 将LLaMA-7B的模型量化为4-bit（推荐）
```bash
python llama_quant.py decapoda-research/llama-7b-hf c4 --wbits 4 --groupsize 128 --save pyllama-7B4b.pt
```

- 将LLaMA-7B的模型量化为2-bit
```bash
python llama_quant.py decapoda-research/llama-7b-hf c4 --wbits 2 --save pyllama-7B2b.pt
```
### 3. 使用gradio推理，你可以直接在网页上操作：
```bash
CUDA_VISIBLE_DEVICES=0
python generate_quant.py \
    --model_path "decapoda-research/llama-7b-hf" \
    --quant_path "pyllama-7B2b.pt" \
    --wbits 2
```

LLaMA_7B量化为2bit后，在beam_search设置为1下推理只需要3.8GB GPU内存。

---


TODO:
- [ ] 调整`merge_lora.py`占用空间过大的问题。
- [ ] 修复由于原代码中的`n_ctx'而导致的分段错误。
- [ ] 加速cpu推理。