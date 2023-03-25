import time

import torch
import torch.nn as nn

from gptq import (
    GPTQ,
    Quantizer,
    find_layers,
    make_quant,
    QuantLinear,
    get_loaders,
    quantize,
)

from quant import LLaMAForCausalLM, LLaMATokenizer, LLaMAConfig
from quant.utils import avoid_tensor_modified, get_llama


@torch.no_grad()
def llama_sequential(model, dataloader, args, dev):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # print("kwargs:", kwargs.keys())
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            i = batch[0].to(dev)
            model(i)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        name_to_gptq = {}
        for name in subset:
            name_to_gptq[name] = GPTQ(subset[name])
            name_to_gptq[name].quantizer = Quantizer()
            name_to_gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=False, mse=False
            )

        def add_batch(name):
            def tmp(_, inp, out):
                name_to_gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
        print(f"\nQuantize layer: {i} ", end=',')
        for name in subset:
            print(name, end=",")
            name_to_gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize)
            quantizers["model.layers.%d.%s" % (i, name)] = name_to_gptq[name].quantizer
            name_to_gptq[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del name_to_gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, args, dev):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


# TODO: perform packing on GPU
def llama_pack(model, quantizers, wbits):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, wbits)
    qlayers = find_layers(model, [QuantLinear])
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    return model


def load_quant(model_name, checkpoint, wbits, seqlen=1024, for_infer=True):
    """
    seqlen - seqlen refers to the maximum length of the input sequence that the model can process. The input sequence can be a sequence of words, tokens, or characters, depending on how the model is tokenized. The seqlen parameter is important because it determines the amount of memory that the model requires to process the input sequence. If the input sequence is too long, it may exceed the memory capacity of the model, leading to out-of-memory errors or slower inference times. In order to handle longer sequences, some models use techniques such as attention masking or truncation, which allow the model to process only a portion of the input sequence at a time. The seqlen parameter determines the maximum length of the input sequence that can be processed in a single step. If the input sequence is longer than the seqlen parameter, it may need to be split into multiple segments and processed separately.
    """
    import transformers

    config = LLaMAConfig.from_pretrained(model_name)
    avoid_tensor_modified()

    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LLaMAForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if for_infer:
        model = model.eval()
    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits)

    print(f"⌛️ Loading model from {checkpoint}...")
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = seqlen
    print(f"✅ Model from {checkpoint} is loaded successfully.")

    return model


def llama_multigpu(model, gpus):
    """A model parallelism implementation for LLaMA"""
    import math
    import copy

    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, "norm") and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[-1])

    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {"mask": None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache["mask"] is None or cache["mask"].device != self.dev:
                cache["mask"] = kwargs["attention_mask"].to(self.dev)
            kwargs["attention_mask"] = cache["mask"]
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus


def run_benchmark(model, input_ids, check=False, dev=torch.device("cuda:0")):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else dev)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=dev)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape(-1),
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            if check and i != input_ids.numel() - 1:
                tot += loss(
                    out.logits[0].to(dev), input_ids[:, (i + 1)].to(dev)
                ).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()
        import numpy as np

        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
            print("max memory(MiB):", max_memory)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="llama model to load",
        default="decapoda-research/llama-7b-hf",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--ckpt_dir", type=str, default="/llama_data/7B")
    parser.add_argument(
        "--tokenizer_path", type=str, default="/llama_data/tokenizer.model"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--nearest", action="store_true", help="Whether to run the RTN baseline."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save quantized checkpoint under this name, eg pyllama-7B4b.pt.",
    )
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Number of tokens to use for benchmarking.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to compute perplexity during benchmarking for verification.",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="cuda:0",
        help="GPU device string, 'cuda:0' by default.",
    )
    parser.add_argument(
        "--eval",
        action="store_false",
        help="Evaluate the model with dataset wikitext2, ptb and c4",
    )

    args = parser.parse_args()
    return args


def run(args=None):
    args = args or get_args()
    if args.load:
        model = load_quant(args.model, args.load, args.wbits)
    else:
        model = get_llama(args.model)
        model.eval()
    if args.cuda.startswith("cuda"):
        dev = torch.device(args.cuda)
    else:
        dev = torch.device("cpu")

    tokenizer = LLaMATokenizer.from_pretrained(
        args.model, add_eos_token=True
    )
    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        tokenizer=tokenizer
    )

    if not args.load and args.wbits < 16 and not args.nearest:
        quantizers = llama_sequential(model, dataloader, args, dev)

    if args.benchmark:
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus)
        else:
            model = model.to(dev)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, : args.benchmark]
            run_benchmark(model, input_ids, check=args.check)
    if args.load:
        exit()

    if args.save:
        llama_pack(model, quantizers, args.wbits)
        torch.save(model.state_dict(), args.save)

    if args.eval:
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, tokenizer=tokenizer
            )
            print(dataset)
            llama_eval(model, testloader, args, dev)


if __name__ == "__main__":
    run()
