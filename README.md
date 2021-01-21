# MLE-Guided Parameter Search (MGS)

PyTorch implementation of the paper:

[MLE-Guided Parameter Search for Task Loss Minimization in Neural Sequence
Modeling](https://arxiv.org/pdf/2006.03158.pdf)\
Sean Welleck, Kyunghyun Cho\
AAAI 2021

## Installation

```bash
python setup.py develop
```

## Data
For downloading the datasets below, it may be helpful to use [gdown.pl](https://github.com/circulosmeos/gdown.pl).

- [Wikitext-103 (sequence-level)](https://drive.google.com/file/d/17nCCBQMVT2dieFR9TD4lNG7cHVxssLsA/view?usp=sharing)

## Pretrained Models
We provide an example base MLE model and example models finetuned with MGS, PG, and MRT.\
Note that metrics in the paper were computed using 5 models per method, each initialized with a different random seed.

| Method |
| --- | 
| [MLE](https://drive.google.com/file/d/1qC6B6JmmqSvKhUQ3z1Xm3ZpZJbdsrhCl/view?usp=sharing) | 
| [MGS-LM](https://drive.google.com/file/d/1OH5UWkfKaAXyc5N2flwRb6ZWh9zjX7ze/view?usp=sharing) | 
| [MGS-LM (ancestral)](https://drive.google.com/file/d/1pYusQkcDBtFEJ2mpnJplS4aDJ7-pRkIA/view?usp=sharing) | 
| [PG-LM](https://drive.google.com/file/d/1rs3fN_MEmjU38K6gtgUewYrQNHoW7L2Z/view?usp=sharing) | 
| [MRT-LM](https://drive.google.com/file/d/1qezjOVB0DH3WPhqwEFcERwhFfNU0vWQ1/view?usp=sharing) | 


## Example commands

Below we show example commands for each stage of the pipeline.\
The experiments in the paper were run with a script external to this repository. 


#### Finetune starting from MLE finetune
```bash
# MGS
python seq_level/gpt2/train.py \
  --loss ggs \
  --ggs-metric lm \
  --ggs-beta 1.0 \
  --model-load-dir /path/to/mle_model

# PG
python seq_level/gpt2/train.py \
  --loss pg \
  --ggs-metric lm \
  --pg-normalize-distance 1 \
  --pg-mle-mix 0.1 \
  --pg-baseline avg \
  --model-load-dir /path/to/mle_model
  
# MRT
python seq_level/gpt2/train.py \
  --loss mrt \
  --ggs-metrc lm \
  --mrt-normalize-distance 1 \
  --mrt-mle-mix 0.1 \
  --model-load-dir /path/to/mle_model
```


#### Finetune MLE
```bash
python seq_level/gpt2/train.py \
  --loss mle \
  --valid-every 5000 \
  --print-every 100
```

#### Evaluate
```bash
python seq_level/gpt2/train.py --mode eval \
  --eval-split valid \ # | test
  --score-model-load-dir /path/to/mle_model \
  --model-load-dir /path/to/model \
  --eval-decoder greedy \ # | temp-1.0
  --token-limit-eval 500 \
  --eval-decode-max-length 500 \
  --chunk-size-valid 512 \
  --loss ggs \
  --ggs-metric lm \
```


#### Preprocess raw wikitext
*not needed if you download the dataset above
```bash
python seq_level/gpt2/prepare_wikitext.py --data-dir /path/to/wikitext-raw
```
