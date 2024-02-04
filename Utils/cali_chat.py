"""
Input: a target model + a text_only or text2text json file
Output: 
    temp = {
            "text": input tokens [512]
            "top_token_prob": top n {tokens:probs} [512, n] -> note that the probs doesn't taking log
            "attention_mask": attention mask [512]
            "loss_mask": loss_mask [512]
    }

"""
import argparse
import datetime
import json
import jsonlines
import torch
import numpy as np
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

def to_tokens_and_logprobs(model, tokenizer, input_text):
    # tokenized_input = tokenizer.encode("###Human: " + input_text['input'] + "###Assistant:")
    tokenized_input = tokenizer.encode(input_text['input'])
    start_idx = len(tokenized_input)
    # tokenized_text = tokenizer.encode("###Human: " + input_text['input'] + "###Assistant:" + input_text['output'])
    tokenized_text = tokenizer.encode(input_text['input'] + input_text['output'])
    tokenized_text_tensor = torch.Tensor([tokenized_text]).to(torch.int32) # [1, 512]
    outputs = model(tokenized_text_tensor)

    probs = torch.softmax(outputs.logits[0, start_idx-1:-1, :], dim=-1).detach() # [1, 512, 32000] - > [output, 32000]
    instances = []
    for i, prob in enumerate(probs): # [32000]
        top5_token = torch.topk(prob, 5).indices.tolist()
        top5_prob = torch.topk(prob, 5).values.tolist()
        ground_truth_token = tokenized_text[start_idx + i]
        ground_truth_prob = prob[ground_truth_token].item()
        temp = {
            "top5_token": top5_token,
            "top5_word": tokenizer.decode(top5_token),
            "top5_prob": top5_prob,
            "ground_truth_token": ground_truth_token,
            "ground_truth_word": tokenizer.decode(ground_truth_token),
            "ground_truth_prob": ground_truth_prob,
        }
        instances.append(temp)
    return instances

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM-Distill")
    parser.add_argument("--model_path", type=str, default="llama33b", help="teacher model path")
    parser.add_argument("--output_dir", type=str, default="/home/mxubh/GSM8K/alpaca_zs_instruction/generated/part.jsonl", help="where to save the generated data")
    parser.add_argument("--dataset_path", type=str, default="", help="path of dataset")
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()    

    print("--- Start Loading Model ---")
    print(f"The model is {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 offload_folder="offload",
                                                 offload_state_dict=True,
                                                 device_map="auto")


    print("--- Start Generating Probability ---")
    print(f"The datset is {args.dataset_path}")
    starttime = datetime.datetime.now()
    with open(args.dataset_path, "r") as data_file:
        data_obj = json.loads(data_file.read())
        data_used = data_obj["instances"]
        total_len = len(data_used)
        for i, input_text in enumerate(data_used):
            output = to_tokens_and_logprobs(model, tokenizer, input_text)
            output_writer = jsonlines.open(args.output_dir, "a")
            output_writer.write(output)
            nowtime = datetime.datetime.now()
            total_time = (nowtime-starttime).seconds
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"{i+1}/{total_len}, passed {round(total_time/60, 2)} mins, still need {round(total_time/60*(total_len-i)/(i+1),2)} mins")
    output_writer.close()

if __name__ == "__main__":
    main()