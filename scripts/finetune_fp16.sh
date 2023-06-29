TOT_CUDA="0,1,2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="11451"

DATA_PATH="sample/instruct/data_sample.jsonl" 
OUTPUT_PATH="outs"
MODEL_PATH="decapoda-research/llama-7b-hf"
TEST_SIZE=1
use_zero_offload=1
if [ ${use_zero_offload} == "1" ]
then
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT finetune_fp16.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_path $MODEL_PATH \
    --eval_steps 200 \
    --save_steps 200 \
    --test_size $TEST_SIZE \
    --deepspeed
else
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune_fp16.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_path $MODEL_PATH \
    --eval_steps 200 \
    --save_steps 200 \
    --test_size $TEST_SIZE
fi
