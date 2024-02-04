
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml generate_top_n.py \
#     --top_n 10 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/gsm8k_finetune_llama33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/train_text2text/zs_train_instruction_text2text.json \
#     --start_index 0 \
#     --end_index 3000 \
#     --output_dir /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/generate/raw/part1.jsonl \

# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml generate_top_n.py \
#     --top_n 10 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/gsm8k_finetune_llama33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/train_text2text/zs_train_instruction_text2text.json \
#     --start_index 3000 \
#     --end_index 6000 \
#     --output_dir /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/generate/raw/part2.jsonl \

# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml generate_top_n.py \
#     --top_n 10 \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/gsm8k_finetune_llama33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/train_text2text/zs_train_instruction_text2text.json \
#     --start_index 6000 \
#     --end_index -1 \
#     --output_dir /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/generate/raw/part3.jsonl \