import argparse
from lm_eval import evaluator
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from awq.utils.lm_eval_adaptor import LMEvalAdaptor

def run_eval(model_path, quant_file, device, tasks, task_batch_size, task_n_shot, task_use_pretrained):
    """
    Post quantization: Evaluate perplexity on wikitext with EleutherAI Evaluation Harness
    """
    # Load model
    if task_use_pretrained:
        model = AutoAWQForCausalLM.from_pretrained(model_path)
    else:
        model = AutoAWQForCausalLM.from_quantized(model_path, quant_file, fuse_layers=False)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load adapter
    lm_eval_model = LMEvalAdaptor(model_path, model, tokenizer, device, batch_size=task_batch_size)

    # Evaluate perplexity of quantized model
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks.split(','),
        batch_size=task_batch_size,
        no_cache=True,
        num_fewshot=task_n_shot,
    )

    print(evaluator.make_table(results))

if __name__ == '__main__':
    """
    - Run perplexity of quantized model:
    python eval.py --model_path casperhansen/mistral-7b-instruct-v0.1-awq

    - Run perplexity unquantized FP16 model:
    python eval.py --use_pretrained --model_path lmsys/vicuna-7b-v1.5
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to hf model')
    parser.add_argument('--quant_file', default='', type=str, help='Path to quantized AWQ model file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to load model to')
    parser.add_argument("--use_pretrained", default=False, action='store_true',
                        help="Pass '--use_pretrained' to use a pretrained model running FP16")
    parser.add_argument('--tasks', type=str, default='wikitext', help='Tasks to evaluate. '
                    'Separate tasks by comma for multiple tasks.'
                    'https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_shot', type=int, default=0)
    args = parser.parse_args()

    run_eval(args.model_path, args.quant_file, args.device,
                       args.tasks, args.batch_size, args.n_shot, args.use_pretrained)