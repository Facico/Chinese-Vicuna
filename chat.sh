BASE_MODEL="decapoda-research/llama-7b-hf" #"/model/13B_hf"
LORA_PATH="Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco" #"checkpoint-6000"
USE_LOCAL=0 # 1: use local model, 0: use huggingface model
DEBUG=0
if [[ USE_LOCAL -eq 1 ]]
then
cp ./config-sample/adapter_config.json $LORA_PATH
fi

if [[ DEBUG -eq 1 ]]
then
    jurigged -v chat.py --model_path $BASE_MODEL --lora_path $LORA_PATH --use_local $USE_LOCAL\
    --use_typewriter 1 \
    --share_link 0 
else
CUDA_VISIBLE_DEVIECES=0 python chat.py --model_path $BASE_MODEL --lora_path $LORA_PATH --use_local $USE_LOCAL\
    --use_typewriter 1 \
    --share_link 0 
fi