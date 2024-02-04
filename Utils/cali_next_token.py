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
    # choice_map = {
    #     "a": 29874,
    #     "b": 29890,
    #     "c": 29883,
    #     "d": 29881,
    #     "e": 29872,
    #     "A": 29909,
    #     "B": 29933,
    #     "C": 29907,
    #     "D": 29928,
    #     "E": 29923,
    #     "no": 694,
    #     "yes": 4874,
    # }

    tokenized_text = tokenizer.encode(input_text['input'])
    tokenized_text_tensor = torch.Tensor([tokenized_text]).to(torch.int32)
    outputs = model(tokenized_text_tensor)
    probs = torch.softmax(outputs.logits[0, -1, :], dim=-1).detach() # [1, 512, 32000] -> [32000]
    text = tokenized_text[-1]

    top5_tokens = torch.topk(probs, 5).indices.tolist()
    top5_probs = torch.topk(probs, 5).values.tolist()

    # probs_yes = probs[choice_map['yes']].item()
    # probs_no = probs[choice_map['no']].item()

    # temp = {
    #         "text": text,
    #         "answer_probs":{
    #             "yes": probs_yes,
    #             "no": probs_no,
    #         },
    #         "top5_probs": top5_probs,
    #         "top5_tokens": top5_tokens,
    #         "top5_words": tokenizer.decode(top5_tokens),
    # }

    # probs_A = probs[choice_map['A']].item()
    # probs_B = probs[choice_map['B']].item()
    # probs_C = probs[choice_map['C']].item()
    # probs_D = probs[choice_map['D']].item()


    temp = {
            "text": text,
            # "answer_probs":{
            #     "A": probs_A,
            #     "B": probs_B,
            #     "C": probs_C,
            #     "D": probs_D,
            # },
            "top5_probs": top5_probs,
            "top5_tokens": top5_tokens,
            "top5_words": tokenizer.decode(top5_tokens),
    }


    return temp

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM-Distill")
    parser.add_argument("--block_size", type=int, default=512, help="fix the size for each input text")
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
    with open(args.dataset_path, "r+") as data_file:
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