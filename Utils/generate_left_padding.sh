CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml generate_left_padding.py \
    --top_n 5 \
    --block_size 512 \
    --model_path pinkmanlove/llama-7b-hf \
    --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train-90/train-90.json \
    --start_index 0 \
    --end_index 1000 \
    --output_dir ./raw/part1.jsonl \
    &
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml generate_left_padding.py \
    --top_n 5 \
    --block_size 512 \
    --model_path pinkmanlove/llama-7b-hf \
    --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train-90/train-90.json \
    --start_index 1000 \
    --end_index -1 \
    --output_dir ./raw/part2.jsonl \