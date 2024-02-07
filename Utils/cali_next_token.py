import argparse
import datetime
import json
import jsonlines
import torch
import numpy as np
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

def to_tokens_and_logprobs(model, tokenizer, input_text, top_n):

    tokenized_text = tokenizer.encode(input_text['input'])
    tokenized_text_tensor = torch.Tensor([tokenized_text]).to(torch.int32).to(model.device)
    outputs = model(tokenized_text_tensor)
    probs = torch.softmax(outputs.logits[0, -1, :], dim=-1).detach() # [1, 512, 32000] -> [32000]

    top_n_tokens = torch.topk(probs, top_n).indices.tolist()
    top_n_probs = torch.topk(probs, top_n).values.tolist()

    temp = {
            "top5_probs": top_n_probs,
            "top5_tokens": top_n_tokens,
            "top5_words": tokenizer.decode(top_n_tokens),
    }

    return temp

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM-Distill")
    parser.add_argument("--top_n", type=int, default=5, help="top-n soft labels")
    parser.add_argument("--model_path", type=str, default="pinkmanlove/llama-7b-hf", help="name or path to model")
    parser.add_argument("--output_dir", type=str, default="./cali_next_token.jsonl", help="where to save the generated data")
    parser.add_argument("--dataset_path", type=str, help="path to dataset")
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
            output = to_tokens_and_logprobs(model, tokenizer, input_text, args.top_n)

            output_writer = jsonlines.open(args.output_dir, "a")
            output_writer.write(output)
            nowtime = datetime.datetime.now()
            total_time = (nowtime-starttime).seconds
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"{i+1}/{total_len}, passed {round(total_time/60, 2)} mins, still need {round(total_time/60*(total_len-i)/(i+1),2)} mins")
    output_writer.close()

if __name__ == "__main__":
    main()