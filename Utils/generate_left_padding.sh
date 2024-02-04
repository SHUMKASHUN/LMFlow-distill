
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml generate_left_padding.py \
#     --top_n 2 \
#     --block_size 128 \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_ao_to_7b_top2_8e-6_norm_left/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/answer_only/test_next_token/test_next_token.json \
#     --start_index 0 \
#     --end_index 700 \
#     --output_dir /home/ksshumab/minrui/StrategyQA/raw/part1.json \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
#     --top_n 2 \
#     --block_size 128 \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_ao_to_7b_top2_8e-6_norm_left/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/answer_only/test_next_token/test_next_token.json \
#     --start_index 700 \
#     --end_index 1400 \
#     --output_dir /home/ksshumab/minrui/StrategyQA/raw/part2.json \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml generate_left_padding.py \
#     --top_n 2 \
#     --block_size 128 \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_ao_to_7b_top2_8e-6_norm_left/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/answer_only/test_next_token/test_next_token.json \
#     --start_index 1400 \
#     --end_index -1 \
#     --output_dir /home/ksshumab/minrui/StrategyQA/raw/part3.jsonl \



# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 128 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_33b_answer_only_8e-6/checkpoint-51 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/answer_only/train_text2text/train_text2text.json \
#     --start_index 0 \
#     --end_index 700 \
#     --output_dir /home/ksshumab/minrui/StrategyQA/raw/part1.json \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 128 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_33b_answer_only_8e-6/checkpoint-51 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/answer_only/train_text2text/train_text2text.json \
#     --start_index 700 \
#     --end_index 1400 \
#     --output_dir /home/ksshumab/minrui/StrategyQA/raw/part2.json \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 128 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_33b_answer_only_8e-6/checkpoint-51 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/answer_only/train_text2text/train_text2text.json \
#     --start_index 1400 \
#     --end_index -1 \
#     --output_dir /home/ksshumab/minrui/StrategyQA/raw/part3.jsonl \



# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml generate_left_padding.py \
#     --top_n 2 \
#     --block_size 128 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_33b_answer_only_8e-6 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/answer_only/train_text2text/train_text2text.json \
#     --start_index 0 \
#     --end_index 700 \
#     --output_dir  /home/ksshumab/minrui/StrategyQA/raw/left/part1.json \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
#     --top_n 2 \
#     --block_size 128 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_33b_answer_only_8e-6 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/answer_only/train_text2text/train_text2text.json \
#     --start_index 700 \
#     --end_index 1400 \
#     --output_dir  /home/ksshumab/minrui/StrategyQA/raw/left/part2.json \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml generate_left_padding.py \
#     --top_n 2 \
#     --block_size 128 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_33b_answer_only_8e-6 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/answer_only/train_text2text/train_text2text.json \
#     --start_index 1400 \
#     --end_index -1 \
#     --output_dir /home/ksshumab/minrui/StrategyQA/raw/left/part3.jsonl \



# gsm8k
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/gsm8k_finetune_33b_zs_1e-5 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/train_text2text/zs_train_instruction_text2text.json \
#     --start_index 0 \
#     --end_index 2600 \
#     --output_dir  /home/ksshumab/minrui/GSM8K/raw/part1.json \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/gsm8k_finetune_33b_zs_1e-5 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/train_text2text/zs_train_instruction_text2text.json \
#     --start_index 2600 \
#     --end_index 5200 \
#     --output_dir  /home/ksshumab/minrui/GSM8K/raw/part2.json \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/gsm8k_finetune_33b_zs_1e-5 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/train_text2text/zs_train_instruction_text2text.json \
#     --start_index 5200 \
#     --end_index -1 \
#     --output_dir /home/ksshumab/minrui/GSM8K/raw/part3.jsonl \


# strategyQA
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml generate_left_padding.py \
#     --top_n 2 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_33b_cot_1e-5 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/train/train.json \
#     --start_index 0 \
#     --end_index 950 \
#     --output_dir /home/ksshumab/minrui/StrategyQA/raw/part1.json \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
#     --top_n 2 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_33b_cot_1e-5 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/train/train.json \
#     --start_index 950 \
#     --end_index -1 \
#     --output_dir /home/ksshumab/minrui/StrategyQA/raw/part2.json \


# OBQA
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml generate_left_padding.py \
#     --top_n 4 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/OBQA_finetune_33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/train/train.json \
#     --start_index 0 \
#     --end_index 1700 \
#     --output_dir  /home/ksshumab/minrui/OBQA/raw/part1.json \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
#     --top_n 4 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/OBQA_finetune_33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/train/train.json \
#     --start_index 1700 \
#     --end_index 3400 \
#     --output_dir  /home/ksshumab/minrui/OBQA/raw/part2.json \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml generate_left_padding.py \
#     --top_n 4 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/OBQA_finetune_33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/train/train.json \
#     --start_index 3400 \
#     --end_index -1 \
#     --output_dir /home/ksshumab/minrui/OBQA/raw/part3.jsonl \




# alpaca
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./data_generation.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca_finetune_33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train_text2text/train.json \
#     --start_index 0 \
#     --end_index 17000 \
#     --temp 0.8 \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/raw/part1_0.8.json \
#     &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca_finetune_33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train_text2text/train.json \
#     --start_index 17000 \
#     --end_index 34000 \
#     --temp 0.8 \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/raw/part2_0.8.json \
#     &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file ./data_generation3.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca_finetune_33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train_text2text/train.json \
#     --start_index 34000 \
#     --end_index -1 \
#     --temp 0.8 \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/raw/part3_0.8.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca_finetune_33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train_text2text/train.json \
#     --start_index 0 \
#     --end_index 26000 \
#     --temp 0.6 \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/raw/part1_0.6.json \
#     &
# CUDA_VISIBLE_DEVICES=8,9 accelerate launch --config_file ./data_generation3.yaml generate_left_padding.py \
#     --top_n 5 \
#     --block_size 512 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca_finetune_33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train_text2text/train.json \
#     --start_index 26000 \
#     --end_index -1 \
#     --temp 0.6 \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/raw/part2_0.6.jsonl \


CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml generate_left_padding.py \
    --top_n 5 \
    --block_size 512 \
    --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama13b-finetune-1e-5-text2text-90 \
    --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train-90/train-90.json \
    --start_index 24000 \
    --end_index 30000 \
    --output_dir /home/ksshumab/minrui/LMFlow-distill/raw/part3.json \
    &
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
    --top_n 5 \
    --block_size 512 \
    --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama13b-finetune-1e-5-text2text-90 \
    --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train-90/train-90.json \
    --start_index 30000 \
    --end_index 36000 \
    --output_dir /home/ksshumab/minrui/LMFlow-distill/raw/part4.jsonl \
    &
CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml generate_left_padding.py \
    --top_n 5 \
    --block_size 512 \
    --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama13b-finetune-1e-5-text2text-90 \
    --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train-90/train-90.json \
    --start_index 36000 \
    --end_index 42000 \
    --output_dir /home/ksshumab/minrui/LMFlow-distill/raw/part5.jsonl \
    &
CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file ./data_generation4.yaml generate_left_padding.py \
    --top_n 5 \
    --block_size 512 \
    --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama13b-finetune-1e-5-text2text-90 \
    --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train-90/train-90.json \
    --start_index 42000 \
    --end_index -1 \
    --output_dir /home/ksshumab/minrui/LMFlow-distill/raw/part6.jsonl \
