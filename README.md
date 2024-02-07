## Table of Contents
- [LMFlow-distill](#lmflow-distill)
  - [Setup](#setup)
  - [Generate](#generate)
  - [Distill](#distill)
    - [Linear Normalization](#linear-normalization)
    - [Softmax Normalization](#softmax-normalization)
    - [Label Smoothing](#label-smoothing)
    - [Other Token](#other-token)
# LMFlow-distill
## Setup
## Generate
## Distill
### Linear Normalization
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/default_config.yaml distill.py \
    --teacher_generation_dataset_path /path/to/teacher/dataset \
    --student_name /path/to/model \
    --output_dir /path/to/output/dir/ \
    --method forward_kl_text2text \
    --use_norm linear \
    --norm_epsilon 1e-6 \
```
### Softmax Normalization
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/default_config.yaml distill.py \
    --teacher_generation_dataset_path /path/to/teacher/dataset \
    --student_name /path/to/model \
    --output_dir /path/to/output/dir/ \
    --method forward_kl_text2text \
    --use_norm softmax \
    --student_temp 1.0 \
    --teacher_temp 1.0 \
```
### Label Smoothing
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/default_config.yaml distill.py \
    --teacher_generation_dataset_path /path/to/teacher/dataset \
    --student_name /path/to/model \
    --output_dir /path/to/output/dir/ \
    --method forward_kl_text2text \
    --use_norm linear \
    --norm_epsilon 1e-6 \
    --user_label_smoothing yes \
    --smoothing_factor 0.1 \
```
### Other Token
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/default_config.yaml distill.py \
    --teacher_generation_dataset_path /path/to/teacher/dataset \
    --student_name /path/to/model \
    --output_dir /path/to/output/dir/ \
    --method forward_kl_text2text \
    --use_other_token yes \
```