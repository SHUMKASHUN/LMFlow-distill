# Partition the whole dataset into 2 parts
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml generate_top_n.py \
    --model_name llama33b \
    --dataset_name gsm8k \
    --start_index 0 \
    --end_index 4000 \
    --dataset_type text2text \
    --output_dir /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/generate/part1.jsonl \
    &
CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml generate_top_n.py \
    --model_name llama33b \
    --dataset_name gsm8k \
    --start_index 4000 \
    --end_index -1 \
    --dataset_type text2text \
    --output_dir /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/generate/part2.jsonl \