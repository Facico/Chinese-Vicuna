DATA_PATH="legal_2048.jsonl"
lora_checkpoint="Chinese-Vicuna/outs/7b-sharegpt-4090-2/train_4800_args"
MODEL_PATH="/model/yahma_llama_7b"
OUTPUT_PATH="outs/7b-legal-from-chatv1-epoch3"

python finetune_chat.py \
--data_path $DATA_PATH \
--model_path $MODEL_PATH \
--output_path $OUTPUT_PATH \
--micro_batch 6 \
--total_batch 32 \
--log_steps 100 \
--eval_steps 0 \
--warmup_ratio 0.01 \
--save_steps 200 \
--test_size 0 \
--prompt_type "chat" \
--resume_from_checkpoint $lora_checkpoint \
--ignore_data_skip True