DATA_PATH="instruct_chat_50k.jsonl"
OUTPUT_PATH="outs/instruct_chat_50k"
MODEL_PATH="yahma/llama-7b-hf"

python finetune_chat.py \
--data_path $DATA_PATH \
--model_path $MODEL_PATH \
--output_path $OUTPUT_PATH \
--micro_batch 3 \
--total_batch 32 \
--log_steps 100 \
--eval_steps 0 \
--warmup_ratio 0.01 \
--save_steps 200 \
--test_size 0 \
--prompt_type "chat"