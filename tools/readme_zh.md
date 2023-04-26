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
提供了一种定量的方法，可以在显存小于4G的设备上使用LLaMA-7B(2bit)模型进行推理。该量化工具参考之前的研究[GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)。
本地测试使用的transformers版本为4.29.0.dev0。
### 1. 首先需要确保模型为huggingface格式。如果不是，可以通过下面的命令转换:
```bash 
python convert_llama.py --input_dir /model/llama-7b --model_size 7B --output_dir ./llama-hf
```
### 2. 然后进行模型量化，下面分别是量化为8bit、4bit、2bit的方法:
- 将LLaMA-7B的模型量化为8-bit
```bash
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b wikitext2 --wbits 8 --true-sequential --act-order --groupsize 128 --save llama7b-8bit-128g.pt
```

- 将LLaMA-7B的模型量化为4-bit（推荐）
```bash
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b wikitext2 --wbits 4 --true-sequential --act-order --groupsize 128 --save llama7b-4bit-128g.pt
```

- 将LLaMA-7B的模型量化为2-bit
```bash
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b wikitext2 --wbits 2 --true-sequential --act-order --groupsize 128 --save llama7b-2bit-128g.pt
```
### 3. 直接生成结果 or 者使用gradio在网页上操作：
- 根据输入的text推理
```bash
python tools/quant_generate.py --model_path ./llama-hf/llama-7b --quant_path llama7b-4bit-128g.pt --wbits 4 --groupsize 128 --text "the mean of life is"
```
- 使用gradio推理，你可以直接在网页上操作
```bash
python tools/quant_generate.py --model_path ./llama-hf/llama-7b --quant_path llama7b-4bit-128g.pt --wbits 4 --groupsize 128 --gradio
```

#### LLaMA-7B 生成结果和显存占用：
- 8bit[8.5G显存]
```text
the mean of life is 70 years.
the median age at death in a population, regardless if it's male or female?
```
- 4bit[5.4G显存]
```text
the mean of life is 70 years.
the median age at death in africa was about what?
```
- 2bit[testing]
```text
testing
```
---


TODO:
- [ ] 调整`merge_lora.py`占用空间过大的问题。
- [ ] 修复由于原代码中的`n_ctx'而导致的分段错误。
- [ ] 加速cpu推理。