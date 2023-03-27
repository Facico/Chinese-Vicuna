CUDA_VISIBLE_DEVICES=0 python generate_quant.py \
    --model_path "decapoda-research/llama-7b-hf" \
    --quant_path "pyllama-7B2b.pt" \
    --wbits 2
