# Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model

![camel](https://github.com/Facico/Chinese-Vicuna/blob/master/img/camel.png)

This is the repo for the Chinese-Vicuna project, which aims to build and share an instruction-following Chinese LLaMA model which can run on a single Nvidia RTX-2080TI, that why we named this project `Vicuna`, small but strong enough !

The repo contains:
- code for finetune the model 
- code for generation based on trained model

## Overview

-  LLaMA paper: https://arxiv.org/abs/2302.13971v1
-  Self-Instruct paper: https://arxiv.org/abs/2212.10560
-  data generation: https://github.com/LianjiaTech/BELLE and https://guanaco-model.github.io/
-  the first work: https://github.com/tatsu-lab/stanford_alpaca

We currently select the combination of BELLE and Guanaco data as our main training dataset. We will also add more chitchat dataset ( e.g. [LCCC](https://github.com/thu-coai/CDial-GPT) ) to support casual conversation.