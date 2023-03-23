BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="./lora-Vicuna/checkpoint-4000"
cp ./config-sample/adapter_config.json $LORA_PATH
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH
