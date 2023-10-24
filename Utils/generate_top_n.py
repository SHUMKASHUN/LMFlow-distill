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
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

def to_tokens_and_logprobs(model, tokenizer, tokenized_text, attention_mask, loss_mask, accelerator):
    tokenized_text_tensor = torch.Tensor([tokenized_text]).to(torch.int32).to(accelerator.device)
    attention_mask_tensor = torch.Tensor([attention_mask]).to(torch.int8).to(accelerator.device)
    outputs = model(input_ids=tokenized_text_tensor, attention_mask=attention_mask_tensor)
    probs = torch.softmax(outputs.logits, dim=-1).detach() # [1, 512, 32000]
    top_token_prob = []
    for j in range(0, probs.shape[1]):
        temp_token = torch.topk(probs[0][j].flatten(), 5).indices.tolist() # [450,   396,  8778,   917, 29871]
        temp_prob = torch.topk(probs[0][j].flatten(), 5).values.tolist() # [0.0520, 0.0459, 0.0347, 0.0297, 0.0270]
        top_token_prob.append({key: value for key, value in zip(temp_token, temp_prob)})
    temp = {
            "text": tokenized_text,
            "top_token_prob": top_token_prob,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
    }
    return temp

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM-Distill")
    parser.add_argument("--top_n", type=int, default=5, help="output the top n probability")
    parser.add_argument("--block_size", type=int, default=512, help="block size")
    parser.add_argument("--model_name", type=str, default="robin33b", help="name of target model")
    parser.add_argument("--output_dir", type=str, default="/home/mxubh/GSM8K/alpaca_zs_instruction/generated/part.jsonl", help="dataset path")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="name of dataset")
    parser.add_argument("--dataset_type", type=str, default="text2ext", choices=["text_only", "text2text"], help="dataset type")
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    model_dict = {
        "llama33b": "pinkmanlove/llama-33b-hf",
        "llama7b": "pinkmanlove/llama-7b-hf",
    }
    dataset_dict = {
        "gsm8k": "/home/ksshumab/minrui/GSM8K/Alpaca_zs_instruction/train_text2text/zs_train_instruction_text2text.json",
    }

    print("--- Start Loading Model ---")
    print(f"The model is {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_dict[args.model_name], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_dict[args.model_name],
                                                 torch_dtype=torch.bfloat16,
                                                 offload_folder="offload",
                                                 offload_state_dict=True,
                                                 device_map="auto")
    accelerator = Accelerator()

    def tokenize_dataset(input_type, data):
        if input_type == "text2text":
            tokenized_input = tokenizer.encode(data['input'])
            tokenized_output = tokenizer.encode(data['output'])
            tokenized_text = tokenized_input + tokenized_output
            padding_length = args.block_size - len(tokenized_text) # pad tokenized_text
            attention_mask = [1] * len(tokenized_text) + [0] * padding_length
            loss_mask = [0] * len(tokenized_input) + [1] * len(tokenized_output) + [0] * padding_length
            tokenized_text += [tokenizer.pad_token_id] * padding_length
            if(padding_length < 0):
                attention_mask = attention_mask[:args.block_size]
                loss_mask = loss_mask[:args.block_size]
                tokenized_text = tokenized_text[:args.block_size]
            assert len(tokenized_text) == len(loss_mask) == len(attention_mask) == args.block_size
        elif input_type == "text_only":
            tokenized_text = tokenizer.encode(data['text'])
            padding_length = args.block_size - len(tokenized_text) # pad tokenized_text
            tokenized_text += [tokenizer.pad_token_id] * padding_length
            attention_mask = loss_mask = [1] * len(tokenized_text) + [0] * padding_length
            if(padding_length < 0):
                attention_mask = attention_mask[:args.block_size]
                loss_mask = loss_mask[:args.block_size]
                tokenized_text = tokenized_text[:args.block_size]
            assert len(tokenized_text) == len(loss_mask) == len(attention_mask) == args.block_size
        else:
            raise NotImplementedError(f"{input_type} not implemented")
        return tokenized_text, loss_mask, attention_mask

    print("--- Start Generating Probability ---")
    print(f"The datset is {args.dataset_name}")
    starttime = datetime.datetime.now()
    with open(dataset_dict[args.dataset_name], "r+") as data_file:
        data_obj = json.loads(data_file.read())
        data_used = data_obj["instances"]
        # data_used = data_obj["instances"][:4000]
        # data_used = data_obj["instances"][4000:]
        total_len = len(data_used)
        for i, item in enumerate(data_used):
            tokenized_text, loss_mask, attention_mask = tokenize_dataset(data_obj["type"], item)
            output = to_tokens_and_logprobs(model, tokenizer, tokenized_text, attention_mask, loss_mask, accelerator)
            output_writer = jsonlines.open(args.output_dir, "a")
            output_writer.write(output)
            nowtime = datetime.datetime.now()
            total_time = (nowtime-starttime).seconds
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"{i+1}/{total_len}, passed {round(total_time/60, 2)} mins, still need {round(total_time/60*(total_len-i)/(i+1),2)} mins")
    output_writer.close()

if __name__ == "__main__":
    main()