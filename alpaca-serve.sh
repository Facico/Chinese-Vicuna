BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="./lora-Vicuna/checkpoint-final"
CUDA_VISIBLE_DEVICES=0 python ./tools/Alpaca-LoRA-Serve/app.py --base_url $BASE_MODEL --ft_ckpt_url $LORA_PATH --port 4321