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

def to_tokens_and_logprobs(model, tokenized_text, top_n, attention_mask, loss_mask, accelerator):
    outputs = model(input_ids=tokenized_text, attention_mask=attention_mask)
    probs = torch.softmax(outputs.logits, dim=-1).detach() # [1, 512, 32000]
    top_token_prob = []
    for j in range(0, probs.shape[1]):
        temp_token = torch.topk(probs[0][j], top_n).indices.tolist() # [450,   396,  8778,   917, 29871]
        temp_prob = torch.topk(probs[0][j], top_n).values.tolist() # [0.0520, 0.0459, 0.0347, 0.0297, 0.0270]
        top_token_prob.append({key: value for key, value in zip(temp_token, temp_prob)})
    temp = {
            "text": tokenized_text[0].tolist(),
            "top_token_prob": top_token_prob,
            "attention_mask": attention_mask[0].tolist(),
            "loss_mask": loss_mask[0].tolist(),
    }
    return temp

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM-Distill")
    parser.add_argument("--top_n", type=int, default=5, help="output the top n probability of teacher model")
    parser.add_argument("--block_size", type=int, default=512, help="fix the size for each input text")
    parser.add_argument("--model_path", type=str, default="llama33b", help="teacher model path")
    parser.add_argument("--output_dir", type=str, default="/home/mxubh/GSM8K/alpaca_zs_instruction/generated/part.jsonl", help="where to save the generated data")
    parser.add_argument("--dataset_path", type=str, default="", help="path of dataset")
    parser.add_argument("--start_index", type=int, default=0, help="start index for dataset partition")
    parser.add_argument("--end_index", type=int, default=-1, help="end index for dataset partition")
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
    accelerator = Accelerator()

    def tokenize_dataset(input_type, data):
        if input_type == "text2text":
            all_text = data['input'] + data['output']
            tokenized_all = tokenizer(all_text, 
                                        padding="max_length", 
                                        max_length=args.block_size,
                                        truncation=True,
                                        return_tensors="pt")
            tokenized_text = tokenized_all.input_ids # tensor([1,block_size])
            attention_mask = tokenized_all.attention_mask # tensor([1,block_size])

            start_idx = torch.nonzero(attention_mask[0] == 1)[0].item()
            input_len = len(tokenizer.encode(data['input']))
            end_idx = start_idx + input_len-1

            loss_mask = torch.ones(attention_mask.shape) # [1, block_size]
            loss_mask[0][:end_idx] = 0
        else:
            raise NotImplementedError(f"{input_type} not implemented")
        return tokenized_text, loss_mask, attention_mask

    print("--- Start Generating Probability ---")
    print(f"The datset is {args.dataset_path}")
    starttime = datetime.datetime.now()
    with open(args.dataset_path, "r+") as data_file:
        data_obj = json.loads(data_file.read())
        if(args.end_index == -1):
            data_used = data_obj["instances"][args.start_index:]
        else:
            data_used = data_obj["instances"][args.start_index:args.end_index]
        total_len = len(data_used)
        for i, item in enumerate(data_used):
            tokenized_text, loss_mask, attention_mask = tokenize_dataset(data_obj["type"], item)
            output = to_tokens_and_logprobs(model, tokenized_text, args.top_n, attention_mask, loss_mask, accelerator)
            output_writer = jsonlines.open(args.output_dir, "a")
            output_writer.write(output)
            nowtime = datetime.datetime.now()
            total_time = (nowtime-starttime).seconds
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"{i+1}/{total_len}, passed {round(total_time/60, 2)} mins, still need {round(total_time/60*(total_len-i)/(i+1),2)} mins")
    output_writer.close()

if __name__ == "__main__":
    main()