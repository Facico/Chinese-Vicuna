This directory offers tools for Vicuna model to run on CPU (in pure C/C++).
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

TODO:
- [ ] speedup cpu inference.
- [ ] fix segmentation fault error.