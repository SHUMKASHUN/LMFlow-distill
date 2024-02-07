CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation5.yaml cali_chat.py \
    --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama13b-finetune-1e-5-text2text-90 \
    --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/validation-10/validation-10.json \
    --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/alpaca/alpaca-openllama13b-finetune-1e-5-validation-text2text-90.jsonl \