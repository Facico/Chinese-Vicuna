DATA_PATH="instuct_chat_50k.jsonl"
OUTPUT_PATH="outs/instuct_chat_50k"
MODEL_PATH="llama-7b"

TOT_CUDA="0,1,2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune_chat.py \
--data_path $DATA_PATH \
--model_path $MODEL_PATH \
--output_path $OUTPUT_PATH \
--micro_batch 4 \
--total_batch 128 \
--log_steps 100 \
--eval_steps 0 \
--warmup_ratio 0.05 \
--save_steps 200 \
--test_size 0 \
--prompt_type "chat"