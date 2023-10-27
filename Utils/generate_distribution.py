import jsonlines
import csv
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

def to_tokens_and_logprobs(model, tokenizer, input_texts, accelerator):
    input_ids = torch.Tensor([input_texts['input_ids']]).to(torch.int32).to(accelerator.device)
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1)
    return probs

def main():
    experiment_name = "distill7b (robin13b)"
    output_file = open("/home/ksshumab/minrui/Distribution_Match/result_30examples.csv", "a")
    output_writer = csv.writer(output_file)

    base_model_path = "LMFlow/Full-Robin-13b-v2"
    # base_model_path = "/home/share/data/robin-33b"

    # draft_model_path = "/home/ksshumab/minrui/LMFlow-distill/output_dir/chat_33b_to_7b_exp3/epoch_2"
    draft_model_path = "/home/ksshumab/minrui/LMFlow-distill/output_dir/chat_13b_to_7b_exp1/epoch_2"
    # draft_model_path = "LMFLow/Full-Robin-7b-v2"
    # draft_model_path = "pinkmanlove/llama-7b-hf"

    data_path = "/home/ksshumab/minrui/Distribution_Match/Chat_Distribution_Example30.jsonl"
    output_path = "/home/ksshumab/minrui/Distribution_Match/result.csv"

    print("--- Start Loading Base Model ---")
    print(f"The base model is: {base_model_path}")
    print(f"The draft model is: {draft_model_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left")
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 offload_folder="offload",
                                                 offload_state_dict=True,
                                                 device_map="auto")

    print("--- Start Loading Draft Model ---")
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_path, padding_side="left")
    draft_tokenizer.pad_token = draft_tokenizer.eos_token
    draft_tokenizer.pad_token_id = draft_tokenizer.eos_token_id

    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 offload_folder="offload",
                                                 offload_state_dict=True,
                                                 device_map="auto")
    accelerator = Accelerator()

    print("--- Start Loading Input Data ---")
    data = []
    with open(data_path, "r+") as f:
        for item in jsonlines.Reader(f):
            data.append(item)

    print("--- Start Generating Probabilities ---")
    output_loss = [experiment_name]
    for input_text in data:
        base_probs = to_tokens_and_logprobs(base_model, base_tokenizer, input_text, accelerator)
        draft_probs = to_tokens_and_logprobs(draft_model, draft_model, input_text, accelerator)
        # use kl div to compare distribution
        
        loss = F.kl_div(base_probs[0], draft_probs[0], reduction="batchmean", log_target=True).detach()
        output_loss.append(loss.item())

    output_writer.writerow(output_loss)    
    output_file.close()

if __name__ == "__main__":
    main()