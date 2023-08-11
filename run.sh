# zero3 offload + checkpointing
CUDA_VISIBLE_DEVICES=0,2,3,4,5,7,8 accelerate launch --config_file configs/default_config.yaml distill.py \
    --dataset_name_or_path ./datasets/Train/33b_blocksize_512_v2.jsonl \
    --per_device_train_batch_size 80 \
    --num_train_epochs 3 \
    --output_dir "./output_dir/distill_512v2_exp9/" \
    --wandb_name "distill_llama7b_512v2_exp9" \
    --gradient_checkpointing \
    --eps 1e-8 \
    --learning_rate 4e-5