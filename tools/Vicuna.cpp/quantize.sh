#!/usr/bin/env bash
MODEL_PATH="../../lora-Vicuna/checkpoint-3000-with-lora/ckpt/ggml-model-f16.bin"
./quantize "$MODEL_PATH" "${MODEL_PATH/f16/q4_0}" 2
