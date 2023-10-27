#!/bin/bash
CUDA_VISIBLE_DEVICES=8,9 accelerate launch --config_file ./data_generation.yaml generate_distribution.py\
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file ./data_generation.yaml generate_distribution.py\
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file ./data_generation.yaml generate_distribution.py\
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file ./data_generation.yaml generate_distribution.py\
# CUDA_VISIBLE_DEVICES=8,9 accelerate launch --config_file ./data_generation.yaml generate_distribution.py\

