LLAMA_PATH=/model/13B_hf
LORA_PATH=xx/checkpoint-1000
TOKENIZER_PATH=xx/tokenizer.model
python merge_lora.py --model_path $LLAMA_PATH --lora_path $LORA_PATH --out_path $LORA_PATH
python convert_pth_to_ggml.py --dir_model $LORA_PATH --fname_tokenizer $TOKENIZER_PATH