TOT_CUDA="0,1,2,3" #Upgrade bitsandbytes to the latest version to enable balanced loading of multiple GPUs, for example: pip install bitsandbytes==0.39.0
BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="/home/cciip/private/fanchenghao/branch/Chinese-Vicuna/lora-Vicuna/checkpoint-16200" #"Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco" #"./lora-Vicuna/checkpoint-final"
USE_LOCAL=1 # 1: use local model, 0: use huggingface model
TYPE_WRITER=1 # whether output streamly
if [[ USE_LOCAL -eq 1 ]]
then
cp sample/instruct/adapter_config.json $LORA_PATH
fi
CUDA_VISIBLE_DEVICES=${TOT_CUDA} python generate_4bit.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH \
    --use_local $USE_LOCAL \
    --use_typewriter $TYPE_WRITER