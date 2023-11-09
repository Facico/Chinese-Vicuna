from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
import argparse

# quant_path = "yahma_llama-7b-hf-awq"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_path",type=str, help="the quantified model path")
    
    parser.add_argument('--text', type=str, default='How are you today?', help='input text')

    parser.add_argument('--min_length', type=int, default=10, help='The minimum length of the sequence to be generated.')

    parser.add_argument('--max_length', type=int, default=512, help='The maximum length of the sequence to be generated.')

    parser.add_argument('--top_p',
                        type=float,
                        default=1.0,
                        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.')

    parser.add_argument('--temperature', type=float, default=1.0, help='The value used to module the next token probabilities.')

    parser.add_argument('--repetition_penalty',type=float, default=1.0, help='The parameter for repetition penalty. 1.0 means no penalty(0~10)')
    args = parser.parse_args()
    # Load model
    quant_path = args.quant_path
    model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
    tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Convert prompt to tokens
    prompt_template = """\
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

    USER: {prompt}
    ASSISTANT:"""

    prompt = args.text

    tokens = tokenizer(
        prompt_template.format(prompt=prompt), 
        return_tensors='pt'
    ).input_ids.cuda()

    # Generate output
    generation_output = model.generate(
        tokens, 
        streamer=streamer,
        min_new_tokens=args.min_length,
        max_new_tokens=args.max_length,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

if __name__ == '__main__':
    main()
