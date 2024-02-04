# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/Data-New/CSQA/model/csqa_2e-5_linear_student_softmax_0.4teacher_temp-90_text2text/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/CSQA/cali_result/distill7b/csqa_2e-5_linear_student_softmax_0.4teacher_temp-90_text2text.jsonl \

# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/Data-New/CSQA/model/csqa_2e-5_linear_student_softmax_0.5teacher_temp-90_text2text/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/CSQA/cali_result/distill7b/csqa_2e-5_linear_student_softmax_0.5teacher_temp-90_text2text.jsonl \

# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/Data-New/CSQA/model/csqa_2e-5_linear_student_softmax_0.6teacher_temp-90_text2text/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/CSQA/cali_result/distill7b/csqa_2e-5_linear_student_softmax_0.6teacher_temp-90_text2text.jsonl \

# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/Data-New/CSQA/model/csqa_2e-5_linear_student_softmax_0.7teacher_temp-90_text2text/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/CSQA/cali_result/distill7b/csqa_2e-5_linear_student_softmax_0.7teacher_temp-90_text2text.jsonl \

# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/Data-New/CSQA/model/csqa_2e-5_linear_student_softmax_0.8teacher_temp-90_text2text/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/CSQA/cali_result/distill7b/csqa_2e-5_linear_student_softmax_0.8teacher_temp-90_text2text.jsonl \

# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/Data-New/CSQA/model/csqa_2e-5_linear_student_softmax_0.9teacher_temp-90_text2text/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/CSQA/cali_result/distill7b/csqa_2e-5_linear_student_softmax_0.9teacher_temp-90_text2text.jsonl \

# CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file ./data_generation.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/csqa_llama2_13b_8e-6-90-text2text \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/CSQA/cali_result/llama2_13b/csqa_llama2_13b_8e-6-test-90-text2text.jsonl \

# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation2.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/csqa_llama2_13b_1e-5-90-text2text \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/validation-10/csqa_validation_test_format.json \
#     --output_dir /home/ksshumab/minrui/Data-New/CSQA/cali_result/llama2_13b/csqa_llama2_13b_1e-5-validation-90-text2text.jsonl \

# CUDA_VISIBLE_DEVICES=8,9 accelerate launch --config_file ./data_generation2.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/Data-New/Alpaca/model/alpaca-llama33b-finetune-1e-5-text2text-90 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/test/test.json \
#     --output_dir /home/ksshumab/minrui/OBQA/cali_result/alpaca-to-obqa/alpaca-llama33b-finetune-1e-5-text2text-90.jsonl \

# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation2.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/Data-New/Alpaca/model/alpaca_llama-7b_linear-2e-5-text2text-90/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/test/test.json \
#     --output_dir /home/ksshumab/minrui/OBQA/cali_result/alpaca-to-obqa/alpaca_llama-7b_linear-2e-5-text2text-90.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=9 accelerate launch --config_file ./data_generation3.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/Data-New/Alpaca/model/alpaca_llama-7b_linear-label_smoothing-2e-5-text2text-90/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/test/test.json \
#     --output_dir /home/ksshumab/minrui/OBQA/cali_result/alpaca-to-obqa/alpaca_llama-7b_linear-label_smoothing-2e-5-text2text-90.jsonl \

# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation2.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama13b-finetune-1e-5-text2text-90 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/test/test.json \
#     --output_dir /home/ksshumab/minrui/OBQA/cali_result/alpaca-to-obqa/alpaca-openllama13b-finetune-1e-5-text2text-90.jsonl \

# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation2.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama13b-finetune-1e-5-text2text-90 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/validation-10/csqa_validation_test_format.json \
#     --output_dir /home/ksshumab/minrui/OBQA/cali_result/alpaca-to-obqa/alpaca-openllama13b-finetune-1e-5-validation-text2text-90.jsonl \

# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation2.yaml cali_next_token.py \
#     --model_path openlm-research/open_llama_7b \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/test/test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/alpaca-to-obqa/obqa-openllama7b-pretrain.jsonl \

# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation5.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama7b-finetune-2e-5-text2text-90 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/test/test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/alpaca-to-obqa/alpaca-openllama7b-finetune-2e-5-text2text-90.jsonl \

# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation5.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_linear_2e-5-text2text-90/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-csqa/alpaca_openllama7b_linear_2e-5-text2text-90.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=9 accelerate launch --config_file ./data_generation4.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_linear_label_smoothing_1e-5-text2text-90/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-csqa/alpaca_openllama7b_linear_label_smoothing_1e-5-text2text-90.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation3.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_linear_label_smoothing_2e-5-text2text-90/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-csqa/alpaca_openllama7b_linear_label_smoothing_2e-5-text2text-90.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=9 accelerate launch --config_file ./data_generation5.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_softmax_0.28_0.28_2e-5-text2text-90/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-csqa/alpaca_openllama7b_softmax_0.28_0.28_2e-5-text2text-90.jsonl \

# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation5.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_softmax_0.28_0.28_3e-5-text2text-90/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-csqa/alpaca_openllama7b_softmax_0.28_0.28_3e-5-text2text-90.jsonl \

# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation4.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama7b-finetune-2e-5-text2text-90 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-csqa/alpaca-openllama7b-finetune-2e-5-text2text-90.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=9 accelerate launch --config_file ./data_generation5.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama13b-finetune-1e-5-text2text-90 \
#     --dataset_path /home/ksshumab/minrui/Data-New/CSQA/test/csqa_test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-csqa/alpaca-openllama13b-finetune-1e-5-text2text-90.jsonl \

# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation4.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama7b-finetune-2e-5-text2text-90 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/test/test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-obqa/alpaca-openllama7b-finetune-2e-5-text2text-90.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation1.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_linear_2e-5-text2text-90/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/test/test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-obqa/alpaca_openllama7b_linear_2e-5-text2text-90.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_next_token.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_linear_label_smoothing_1e-5-text2text-90/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/test/test.json \
#     --output_dir /home/ksshumab/minrui/TemperatureScaling/alpaca-to-obqa/alpaca_openllama7b_linear_label_smoothing_1e-5-text2text-90.jsonl \
#     &
CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file ./data_generation1.yaml cali_next_token.py \
    --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_softmax_0.28_0.28_1e-5-text2text-90/epoch_2 \
    --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/validation-10/validation.json \
    --output_dir /home/ksshumab/minrui/TemperatureScaling/post-cali/alpaca_openllama7b_softmax_0.28_0.28_1e-5-validation-text2text-90.jsonl \
    &
CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation2.yaml cali_next_token.py \
    --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama7b-finetune-2e-5-text2text-90 \
    --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/validation-10/validation.json \
    --output_dir /home/ksshumab/minrui/TemperatureScaling/post-cali/alpaca-openllama7b-finetune-2e-5-validation-text2text-90.jsonl \
    &
CUDA_VISIBLE_DEVICES=5 accelerate launch --config_file ./data_generation3.yaml cali_next_token.py \
    --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_openllama7b_linear_2e-5-text2text-90/epoch_2 \
    --dataset_path /home/ksshumab/minrui/OBQA/csqa_format/validation-10/validation.json \
    --output_dir /home/ksshumab/minrui/TemperatureScaling/post-cali/alpaca_openllama7b_linear_2e-5-validation-text2text-90.jsonl \
