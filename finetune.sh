TOT_CUDA="2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="/home/cciip/private/fanchenghao/dataset/instruction/merge.json" #"./sample/merge_sample.json"
OUTPUT_PATH="lora-Vicuna"
MODEL_PATH="decapoda-research/llama-7b-hf"

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--resume_from_checkpoint './lora-Vicuna/checkpoint-5800' \
--test_size 2000