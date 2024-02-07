
CUDA_VISIBLE_DEVICES=5 accelerate launch --config_file ./data_generation3.yaml cali_next_token.py \
    --top_n 5 \
    --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_linear_2e-5-text2text-90/epoch_2 \
    --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/validation-10/validation.json \
    --output_dir /home/ksshumab/minrui/TemperatureScaling/post-cali/alpaca_openllama7b_linear_2e-5-validation-text2text-90.jsonl \
