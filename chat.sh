BASE_MODEL="/model/13B_hf"
LORA_PATH="checkpoint-6000"
USE_LOCAL=0 # 0: use local model, 1: use huggingface model
DEBUG=0
if [ USE_LOCAL == 1 ]
then
cp ./config-sample/adapter_config.json $LORA_PATH
fi

if [ DEBUG == 1 ]
then
jurigged -v chat.py --model_path $BASE_MODEL --lora_path $LORA_PATH --use_local $USE_LOCAL\
 --use_typewriter 1 \
 --share_link 0 
else
python chat.py --model_path $BASE_MODEL --lora_path $LORA_PATH --use_local $USE_LOCAL\
    --use_typewriter 1 \
    --share_link 0 
fi