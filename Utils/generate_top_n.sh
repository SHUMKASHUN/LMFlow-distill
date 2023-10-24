CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml generate_top_n.py \
    --model_name llama33b \
    --dataset_name gsm8k \
    --dataset_type text2text \
    --output_dir /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/generate/all.jsonl \