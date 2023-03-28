BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco" #"./lora-Vicuna/checkpoint-final"
USE_LOCAL=0 #1 | 0
if [ USE_LOCAL == 1 ]
then
cp ./config-sample/adapter_config.json $LORA_PATH
fi
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH \
    --use_local $USE_LOCAL
