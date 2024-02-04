# ----------------- chat
# /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json
# /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json
# /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json

# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-7b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/single_llama7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-7b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/multi_llama7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-7b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/instruction_llama7b.jsonl \


# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/share/data/robin-33b \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/single_robin33b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/share/data/robin-33b \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/multi_robin33b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/share/data/robin-33b \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/instruction_robin33b.jsonl \


# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path LMFlow/Full-Robin-7b-v2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/single_robin7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path LMFlow/Full-Robin-7b-v2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/multi_robin7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path LMFlow/Full-Robin-7b-v2 \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/instruction_robin7b.jsonl \


# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/chat_33b_to_7b_exp3/epoch_2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/single_distill7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/chat_33b_to_7b_exp3/epoch_2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/multi_distill7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/chat_33b_to_7b_exp3/epoch_2 \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/instruction_distill7b.jsonl \


# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/single_llama33b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/multi_llama33b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/instruction_llama33b.jsonl \


# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/single_llama33b_no_prompt_structure.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/multi_llama33b_no_prompt_structure.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/instruction_llama33b_no_prompt_structure.jsonl \



# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-13b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/single_llama13b_no_prompt_structure.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-13b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/multi_llama13b_no_prompt_structure.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=6 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-13b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/instruction_llama13b_no_prompt_structure.jsonl \


# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-13b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/single_llama13b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-13b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/multi_llama13b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-13b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/instruction_llama13b.jsonl \


# CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-13b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/single_llama13b_no_prompt_structure.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-13b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/single_llama13b.jsonl \



# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-7b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/single_llama7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-7b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/multi_llama7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-7b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/instruction_llama7b.jsonl \


# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/share/data/robin-33b \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/single_robin33b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/share/data/robin-33b \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/multi_robin33b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/share/data/robin-33b \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/instruction_robin33b.jsonl \


# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path LMFlow/Full-Robin-7b-v2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/single_robin7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path LMFlow/Full-Robin-7b-v2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/multi_robin7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path LMFlow/Full-Robin-7b-v2 \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/instruction_robin7b.jsonl \


# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/chat_33b_to_7b_exp3/epoch_2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/single_distill7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/chat_33b_to_7b_exp3/epoch_2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/multi_distill7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/chat_33b_to_7b_exp3/epoch_2 \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/instruction_distill7b.jsonl \


# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/single_llama33b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/multi_llama33b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=6,7,8 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/instruction_llama33b.jsonl \




# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-7b-hf \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/test/zs_test_instruction.json \
#     --output_dir /home/ksshumab/minrui/GSM8K/cali_result/gsm8k_llama7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/gsm8k_finetune_llama7b_2e-5_epoch3 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/test/zs_test_instruction.json \
#     --output_dir /home/ksshumab/minrui/GSM8K/cali_result/gsm8k_finetune7b.jsonl \

# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/gsm8k_llama33b_to_7b_top5_ot_norm_1e-5/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/test/zs_test_instruction.json \
#     --output_dir /home/ksshumab/minrui/GSM8K/cali_result/gsm8k_distill7b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1,2 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/gsm8k_finetune_33b_zs_1e-5_epoch3 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/test/zs_test_instruction.json \
#     --output_dir /home/ksshumab/minrui/GSM8K/cali_result/gsm8k_finetune33b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=3,4 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/test/zs_test_instruction.json \
#     --output_dir /home/ksshumab/minrui/GSM8K/cali_result/gsm8k_llama33b.jsonl \


# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_ao_to_7b_top2_8e-6_left_replace/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/test/zs_test_instruction.json \
#     --output_dir /home/ksshumab/minrui/GSM8K/cali_result/gsm8k_distill7b.jsonl \



# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path LMFlow/Full-Robin-13b-v2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_single_nll_text2text/lmflow_chat_en_dialog_multiturn_single_nll_text2text.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/single_robin13b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path LMFlow/Full-Robin-13b-v2 \
#     --dataset_path /home/ksshumab/LMFlow/data/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp/lmflow_chat_en_dialog_multiturn_nll_text2text_nosharp.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/multi_robin13b.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path LMFlow/Full-Robin-13b-v2 \
#     --dataset_path /home/ksshumab/LMFlow/data/gp4_instruction_en_eval/gpt4_en_sample_v1_1k_2k_split_eval.json \
#     --output_dir /home/ksshumab/minrui/Chat/cali_result/prompt_structure/instruction_robin13b.jsonl \




# -------------- strategyQA with COT

# llama7b
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-7b-hf \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/llama7b/strategyQA_llama_7b_cot.jsonl \

# llama33b
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/llama33b/strategyQA_llama_33b_cot.jsonl \

# finetune7b 1e-5
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_1e-5/checkpoint-15 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_1e-5_cot_epoch0.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_1e-5/checkpoint-30 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_1e-5_cot_epoch1.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_1e-5/checkpoint-45 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_1e-5_cot_epoch2.jsonl \

# finetune7b 2e-5
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_2e-5/checkpoint-15 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_2e-5_cot_epoch0.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_2e-5/checkpoint-30 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_2e-5_cot_epoch1.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_2e-5/checkpoint-45 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_2e-5_cot_epoch2.jsonl \

# finetune7b 3e-5
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_3e-5/checkpoint-15 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_3e-5_cot_epoch0.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_3e-5/checkpoint-30 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_3e-5_cot_epoch1.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_3e-5/checkpoint-45 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_3e-5_cot_epoch2.jsonl \


# finetune7b 1e-5 A
# CUDA_VISIBLE_DEVICES=9 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_7b_cot_1e-5_A \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune7b/strategyQA_finetune_7b_1e-5_A_cot.jsonl \

# finetune33b 1e-5
# CUDA_VISIBLE_DEVICES=7,8,9 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_33b_cot_1e-5 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune33b/strategyQA_finetune_33b_1e-5_cot.jsonl \

# distill7b strategyQA_llama33b_1e-5_to_7b_top2_1e-5_softmax
# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./cali.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_1e-5_to_7b_top2_1e-5_softmax/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/distill7b/strategyQA_distill_7b_1e-5_softmax_cot.jsonl \

# distill7b strategyQA_llama33b_1e-5_to_7b_top2_2e-5_softmax
# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./cali.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_1e-5_to_7b_top2_2e-5_softmax/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/distill7b/strategyQA_distill_7b_2e-5_softmax_cot.jsonl \

# distill7b strategyQA_llama33b_1e-5_to_7b_top2_1e-5_linear
# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./cali.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_1e-5_to_7b_top2_1e-5_linear/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/distill7b/strategyQA_distill_7b_1e-5_linear_cot.jsonl \

# distill7b strategyQA_llama33b_1e-5_to_7b_top2_1e-5_linear replace
# CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./cali.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_1e-5_to_7b_top2_1e-5_linear_replace/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/distill7b/strategyQA_distill_7b_1e-5_linear_replace_cot.jsonl \

# distill7b strategyQA_llama33b_1e-5_to_7b_top2_2e-5_softmax
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_1e-5_to_7b_top2_8e-6_softmax/epoch_0 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/distill7b/strategyQA_distill_7b_8e-6_softmax_cot_epoch0.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_1e-5_to_7b_top2_8e-6_softmax/epoch_1 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/distill7b/strategyQA_distill_7b_8e-6_softmax_cot_epoch1.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_1e-5_to_7b_top2_8e-6_softmax/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/distill7b/strategyQA_distill_7b_8e-6_softmax_cot_epoch2.jsonl \

# CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/strategyQA/strategyQA_llama33b_1e-5_to_7b_top2_8e-6_softmax_replace/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/distill7b/strategyQA_distill_7b_8e-6_softmax_replace_cot_epoch2.jsonl \

# llama13b
# CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-13b-hf \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/llama13b/strategyQA_llama_13b_cot.jsonl \


# finetune13b /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_13b_cot_3e-5
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/strategyQA_finetune_13b_cot_3e-5/checkpoint-40 \
#     --dataset_path /home/ksshumab/minrui/StrategyQA/StrategyQA_Cot/test/test.json \
#     --output_dir /home/ksshumab/minrui/StrategyQA/cali_result/finetune13b/strategyQA_finetune_13b_3e-5_cot_epoch1.jsonl \




# alpaca -------------------------
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca_finetune_7b_2e-5_4device \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/test/test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/finetune7b/alpaca_finetune_7b_2e-5_4device.jsonl \
#     &
# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca_finetune_7b_2e-5_8device \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/test/test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/finetune7b/alpaca_finetune_7b_2e-5_8device.jsonl \

# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-7b-hf \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/test/test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/llama7b/alpaca_llama_7b.jsonl \

# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path pinkmanlove/llama-33b-hf \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/test/test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/llama33b/alpaca_llama_33b.jsonl \



# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file ./data_generation4.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca_finetune_33b_1e-5 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/test/test.json \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/finetune33b/alpaca_test_finetune_33b_1e-5.jsonl \

# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./data_generation4.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_33b_1e-5_to_7b_2e-5_top5_softmax_0.25teacher_temp/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/train_text2text/train.json \
#     --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/alpaca_train_distill_7b_softmax_2e-5_temp1.2.jsonl \


# CUDA_VISIBLE_DEVICES=8,9 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_33b_1e-5_to_7b_2e-5_top5_softmax_0.8student_0.4teacher_temp/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/test/test.json \
#     --output_dir /home/ksshumab/minrui/alpaca_distill_7b_softmax_0.8student_0.4teacher_temp_epoch2.jsonl \

# CUDA_VISIBLE_DEVICES=8,9 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_33b_1e-5_to_7b_2e-5_top5_softmax_0.8student_0.6teacher_temp/epoch_2 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/test/test.json \
#     --output_dir /home/ksshumab/minrui/alpaca_distill_7b_softmax_0.8student_0.6teacher_temp_epoch2.jsonl \

# CUDA_VISIBLE_DEVICES=8,9 accelerate launch --config_file ./data_generation2.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_33b_1e-5_to_7b_2e-5_top5_softmax_0.6student_0.4teacher_temp/conti/epoch_1 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/test/test.json \
#     --output_dir /home/ksshumab/minrui/alpaca_distill_7b_softmax_0.6student_0.4teacher_temp_epoch2.jsonl \

# CUDA_VISIBLE_DEVICES=8,9 accelerate launch --config_file ./data_generation3.yaml cali_chat.py \
#     --model_path /home/ksshumab/minrui/LMFlow-distill/output_dir/alpaca/alpaca_33b_1e-5_to_7b_2e-5_top5_softmax_0.6student_0.6teacher_temp/conti/epoch_1 \
#     --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/test/test.json \
#     --output_dir /home/ksshumab/minrui/alpaca_distill_7b_softmax_0.6student_0.6teacher_temp_epoch2.jsonl \

CUDA_VISIBLE_DEVICES=8 accelerate launch --config_file ./data_generation5.yaml cali_chat.py \
    --model_path /home/ksshumab/minrui/new-lmflow/LMFlow/output_models/alpaca-openllama13b-finetune-1e-5-text2text-90 \
    --dataset_path /home/ksshumab/minrui/Data-New/Alpaca/validation-10/validation-10.json \
    --output_dir /home/ksshumab/minrui/Data-New/Alpaca/cali_result/alpaca/alpaca-openllama13b-finetune-1e-5-validation-text2text-90.jsonl \