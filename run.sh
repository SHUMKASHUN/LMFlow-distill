CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./configs/default_config.yaml distill.py\
    --per_device_train_batch_size 36 \
    --dataset_name Test \
