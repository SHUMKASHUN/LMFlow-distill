

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/default_config.yaml distill.py \
    --dataset_name_or_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/generate/llama33b_finetune_1e-5_top5_p.jsonl \
    --per_device_train_batch_size 32 \
    --num_train_epochs 3 \
    --student_name "pinkmanlove/llama-7b-hf" \
    --output_dir "./output_dir/gsm8k_llama33b_to_7b_top3_p/" \
    --wandb_name "gsm8k_llama33b_to_7b_top3_p" \
    --gradient_checkpointing \
    --eps 1e-8 \
    --learning_rate 4e-5 \
    --use_linear_norm 0 \
    --use_other_token 0 \
    --method mixed_text2text \
    --epsilon 1e-6 \

    