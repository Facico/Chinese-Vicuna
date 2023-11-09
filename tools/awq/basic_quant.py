from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import argparse

# model_path = 'models/yahma_llama-7b-hf'
# quant_path = 'yahma_llama-7b-hf-awq'
def quant_model(model_path, quant_path, quant_config):
    # Load model
    # NOTE: pass safetensors=True to load safetensors
    model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default = 'm  ', help='Path to hf model')
    parser.add_argument('--save', default='', type=str, help='Path to quantized AWQ model file')
    parser.add_argument('--q_group_size', default=128, type=int, help='Quantization group size')
    parser.add_argument('--version', default='GEMM', type=str, choices=['GEMM', 'GEMV'], help='GEMM vs GEMV are related to howmatrix multiplication runs under the hood.')
    args = parser.parse_args()

    quant_config={ "zero_point": True, "q_group_size": args.q_group_size, "w_bit": 4, "version": args.version}
    quant_model(args.model_path, args.save, quant_config)