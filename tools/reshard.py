# ref: https://gist.github.com/benob/4850a0210b01672175942203aa36d300
import os
import json
import sys
import torch
import glob

# python test.py 2 xx/checkpoint-1000/ckpt/ outs

if len(sys.argv) != 4:
    print('usage: %s <new-shards> <input-model-path> <output-model-path>' % sys.argv[0], file=sys.stderr)
    sys.exit(1)

num_shards = int(sys.argv[1])
input_model_dir = sys.argv[2]
output_model_dir = sys.argv[3]

with open(os.path.join(input_model_dir, 'params.json'), 'r') as fp:
    params = json.loads(fp.read())

assert params['dim'] % num_shards == 0, "number of shards need to divide parameter dimension %d" % params['dim']

print('loading...')
checkpoints = [torch.load(path, map_location=torch.device('cpu')) for path in glob.glob(os.path.join(input_model_dir, '*.pth'))]

layer_kind = {
    'tok_embeddings': 'ParallelEmbedding',
    'output': 'ColumnParallelLinear',
    'attention.wq': 'ColumnParallelLinear',
    'attention.wk': 'ColumnParallelLinear',
    'attention.wv': 'ColumnParallelLinear',
    'attention.wo': 'RowParallelLinear',
    'feed_forward.w1': 'ColumnParallelLinear',
    'feed_forward.w2': 'RowParallelLinear',
    'feed_forward.w3': 'ColumnParallelLinear',
    'attention_norm': None,
    'ffn_norm': None,
    'norm': None,
    'rope.freqs': None,
}

output = [dict() for x in range(num_shards)]

print('converting...')
for key in checkpoints[0].keys():
    tensors = [m[key] for m in checkpoints]
    print(key)
    print('  in shapes=', [p.shape for p in tensors])
    for pattern, kind in layer_kind.items():
        if key.replace('.weight', '').endswith(pattern):
            print('  kind=', kind)
            if kind == 'ColumnParallelLinear':
                with torch.no_grad():
                    merged = torch.cat(tensors, 0)
                    slice_size = merged.shape[0] // num_shards
                    for rank in range(num_shards):
                        output[rank][key] = merged[slice_size * rank: slice_size * (rank + 1),:].clone().detach()
            elif kind in ('ParallelEmbedding', 'RowParallelLinear'):
                with torch.no_grad():
                    merged = torch.cat(tensors, 1)
                    slice_size = merged.shape[1] // num_shards
                    for rank in range(num_shards):
                        output[rank][key] = merged[:,slice_size * rank: slice_size * (rank + 1)].clone().detach()
            else:
                for rank in range(num_shards):
                    output[rank][key] = tensors[0]
            print('  out shapes=', [output[rank][key].shape for rank in range(num_shards)])
            print()
            break
    else:
        raise Exception('parameter name not recognized')

print('saving...')
os.makedirs(output_model_dir, exist_ok=True)
with open(os.path.join(output_model_dir, 'params.json'), 'w') as fp:
    fp.write(json.dumps(params))

for rank in range(num_shards):
    print(' ', rank)
    torch.save(output[rank], os.path.join(output_model_dir, 'consolidated.%02d.pth' % rank))

print('done.')