BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco" #"./lora-Vicuna/checkpoint-final"
USE_LOCAL=0 # 0: use local model, 1: use huggingface model
if [ USE_LOCAL == 1 ]
then
cp ./config-sample/adapter_config.json $LORA_PATH
fi
CUDA_VISIBLE_DEVICES=2 python interaction.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH \
    --use_local $USE_LOCAL
