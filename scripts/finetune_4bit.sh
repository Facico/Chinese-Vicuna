TOT_CUDA="0,1,2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="sample/instruct/data_sample.jsonl"
OUTPUT_PATH="lora-Vicuna"
MODEL_PATH="/model/yahma_llama_7b"
lora_checkpoint="./lora-Vicuna/checkpoint-11600"
TEST_SIZE=1

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune_4bit.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE
