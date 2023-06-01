TOT_CUDA="0,1,2,3" #Upgrade bitsandbytes to the latest version to enable balanced loading of multiple GPUs, for example: pip install bitsandbytes==0.39.0
BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco" #"./lora-Vicuna/checkpoint-final"
USE_LOCAL=0 # 1: use local model, 0: use huggingface model
if [ ${USE_LOCAL} == "1" ]
then
cp sample/instruct/adapter_config.json $LORA_PATH
fi
CUDA_VISIBLE_DEVICES=${TOT_CUDA} python interaction.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH \
    --use_local $USE_LOCAL
