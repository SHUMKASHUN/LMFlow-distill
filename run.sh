CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./default_config.yaml distill.py\
    --teacher_name text-davinci-002 \
    --student_name pinkmanlove/llama-7b-hf \
     --dataset_name example_dataset
