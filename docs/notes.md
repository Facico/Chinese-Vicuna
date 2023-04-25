1. use specific peft/bitandbytes/transformers package (which lists in our requirements.txt)  : 
- https://github.com/Facico/Chinese-Vicuna/issues/70
- https://github.com/Facico/Chinese-Vicuna/issues/88
- https://github.com/Facico/Chinese-Vicuna/issues/81 

2. better not use V100 & P40 GPUs, which may not support int8 : 
- https://github.com/Facico/Chinese-Vicuna/issues/39


3. don't use torchrun on single cards: 
- https://github.com/Facico/Chinese-Vicuna/issues/4
- https://github.com/Facico/Chinese-Vicuna/issues/45
