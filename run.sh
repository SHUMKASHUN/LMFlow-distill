# zero3 offload + checkpointing
CUDA_VISIBLE_DEVICES=0,2,3,4,5 accelerate launch --config_file configs/default_config.yaml distill.py \
    --per_device_train_batch_size 80 \
    --dataset_name Train \
    --num_train_epochs 1 \
    --output_dir "./output_dir/distill_512v2_exp4/" \
    --wandb_name "distill_llama7b_512v2_exp4" \
    --gradient_checkpointing