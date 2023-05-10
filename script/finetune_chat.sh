TOT_CUDA="1,2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH='sample/merge_sample.json'
OUTPUT_PATH="outs/7B_hf"
MODEL_PATH="/model/7B_hf"

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune_chat.py \
--data_path $DATA_PATH \
--model_path $MODEL_PATH \
--output_path $OUTPUT_PATH \
--micro_batch 2 \
--total_batch 128 \
--log_steps 100 \
--eval_steps 0 \
--warmup_steps 30 \
--save_steps 200 \
--test_size 0 \
