import logging
import sys
import random
import torch
import os
import json
import jsonlines
from tqdm import tqdm
import torch.nn.functional as F
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed
)
from utils import load_special_dataset_for_train
import argparse
from torch.utils.data import DataLoader, Dataset
import math
from accelerate import Accelerator

# Load teach dataset with jsonl file
# {
#     "text": list, #(int)
#     "logprobs": {
#         "tokens": list, #(str)
#         "token_logprobs": list, #(float)
#         "top_log_probs": dict #(str:float)
#     }
# }
# We only need text and top_log_probs
class TeacherDataset(Dataset):
    def __init__(self, data_path: str):
        print("-----Loading Teacher Dataset-----")
        self.data = []
        with open(data_path, "r") as f:
            for item in jsonlines.Reader(f): 
                del item["logprobs"]["tokens"] # delete unused tokens
                del item["logprobs"]["token_logprobs"] # delete unused token_logprobs
                del item["logprobs"]["top_log_probs"][0] # delete starting null value
                self.data.append(item)
        print("-----Finish Loading Teacher Dataset-----")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return{
            'text': self.data[index]['text'],
            'top_log_probs': self.data[index]["logprobs"]['top_log_probs']
        }
    
    def collate_fn(self, batch):
        return{
            'text': [x['text'] for x in batch],
            'top_log_probs': [x['top_log_probs'] for x in batch]
        }

def save_history(chat, chat_history_dir):
    if os.path.exists(f"{chat_history_dir}/chatgpt_distill_history.json"):
        with open(f"{chat_history_dir}/chatgpt_distill_history.json", "r") as f:
            old_data = json.load(f)
            old_data["Instances"].extend(chat)
        with open(f"{chat_history_dir}/chatgpt_distill_history.json", "w") as f:
            json.dump(old_data, f, indent=4) 
    else:
        return_json = {}
        return_json["Contributors"] = "Shizhe Diao"
        return_json["Description"] = "The history of distilling chatgpt to gpt-2",
        return_json["Instances"] = chat
        with open(f"{chat_history_dir}/chatgpt_distill_history.json", "w") as f:
            json.dump(return_json, f, indent=4) 

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM-Distill")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="learning rate")
    parser.add_argument("--dataset_name", type=str, default="chatgpt-prompt", help="dataset name")
    parser.add_argument("--teacher_name", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "code-davinci-002", "text-davinci-002"], help="teacher model name")
    parser.add_argument("--student_name", type=str, default="pinkmanlove/llama-7b-hf", help="student model name") #gpt2
    parser.add_argument("--max_tokens", type=int, default=3, help="minimum length for generation")
    parser.add_argument("--max_num_log_probs", type=int, default=5, help="minimum length for generation")
    parser.add_argument("--stop_token", default=None, help="stop token of GPT-3, choice=[\n, None],")
    parser.add_argument("--teacher_temp", type=float, default=0.7, help="temperature of the teacher")
    parser.add_argument("--student_temp", type=float, default=0.7, help="temperature of the student")
    parser.add_argument("--log_level", type=str, default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="logging level")
    parser.add_argument("--validation_split_percentage", type=int, default=20, help="the percentage of validation split")
    parser.add_argument("--demo_example_in_prompt", type=bool, default=False, help="When this flag is True, the prompt will include examplary, samples in the prompt if available from the dataset.")
    parser.add_argument("--local_rank", type=int, help="local rank")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="train batch size per device") #8
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="eval batch size per device") #8
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", type=int, default=1e10, help="max steps for debug.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="Where to store the final model.")

    args = parser.parse_args()

    return args

def main():
    args = arg_parser()
    max_tokens = args.max_tokens 
    max_num_log_probs = args.max_num_log_probs
    teacher_name = args.teacher_name
    student_name = args.student_name
    stop = args.stop_token
    teacher_temp = args.teacher_temp
    student_temp = args.student_temp
    set_seed(args.random_seed)
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Setup student model
    logger.info("*** [START] Setting up student model ***")
    config = AutoConfig.from_pretrained(student_name)
    tokenizer = AutoTokenizer.from_pretrained(student_name, use_fast=False)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        from_tf=False,
        # config=config,
    )
    logger.info("*** [FINISH] Setting up student model ***")

    # Load dataset
    logger.info("*** [START] Loading dataset ***")
    raw_datasets = load_special_dataset_for_train(
        dataset_name=args.dataset_name,
        validation_split_percentage=args.validation_split_percentage,
        demo_example_in_prompt=args.demo_example_in_prompt,
        local_rank=args.local_rank,
    )
    logger.info(f"Load customized dataset complete, "
                f"training samples {len(raw_datasets['train'])}, "
                f"validation samples {len(raw_datasets['validation'])}."
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    logger.info("*** [FINISH] Loading dataset ***")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    logger.info("*** [START] Creating dataloader ***")

    data_path = './dataset/data_0501/0-120.jsonl'
    teacher_dataset = TeacherDataset(data_path)
    train_dataloader = DataLoader(teacher_dataset, 
                                  batch_size=args.per_device_train_batch_size, 
                                  collate_fn=teacher_dataset.collate_fn)

    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=False, batch_size=args.per_device_train_batch_size
    # )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.per_device_eval_batch_size
    )
    logger.info("*** [FINISH] Creating dataloader ***")

    # Optimizer
    logger.info("*** [START] Setting up optimizer and scheduler ***")
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, tokenizer = accelerator.prepare(
        student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, tokenizer
    )
    logger.info("*** [FINISH] Setting up optimizer and scheduler ***")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    # Distill from soft probs.
    if args.teacher_name in ["text-davinci-002"]:

        for epoch in range(starting_epoch, args.num_train_epochs):
            print("epoch", epoch)
            student_model.train()
            if args.with_tracking:
                total_loss = 0

            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(student_model):
                    batch_loss = 0
                    for i in range(len(batch)): # for each batch

                        # we will use the teacher_model result directly
                        teacher_top_logprobs = batch['top_log_probs'][i]
                        teacher_output_tokens = batch['text'][i]

                        teacher_logprobs_list = []
                        id_list = []
                        for teacher_step in teacher_top_logprobs[-max_tokens:]: # question: why the tail?
                            teacher_logprobs_list.append([])
                            id_list.append([])
                            for token, logprob in teacher_step.items():
                                id = tokenizer.encode(token)[0]
                                teacher_logprobs_list[-1].append(logprob)
                                id_list[-1].append(id)

                        teacher_logprobs_tensor = torch.tensor(teacher_logprobs_list).reshape(-1).to("cuda")
                        teacher_probs_tensor = teacher_logprobs_tensor.exp().float() # convert back to regular prob
                        id_tensor = torch.tensor(id_list)

                        # student model training
                        teacher_batch = torch.Tensor(teacher_output_tokens).to(torch.int32).to("cuda:0") # here, teacher_output_tokens is already encoded

                        ### llama
                        # teacher_batch = teacher_batch.unsqueeze(0)
                        # outputs = student_model(teacher_batch) # student_model: cuda
                        # student_logits = outputs.logits[0][-max_tokens:]
                        ### gpt2
                        outputs = student_model(teacher_batch) # student_model: cuda
                        student_logits = outputs.logits[-max_tokens:] # outputs.logits.shape = [512,50257]

                        student_probs_full = F.softmax(student_logits / student_temp, dim=1) # [3,50257]
                        
                        row_indices = torch.tensor([[i] * max_num_log_probs for i in range(max_tokens)])
                        row_indices = row_indices.reshape(-1)
                        id_tensor = id_tensor.reshape(-1)

                        student_logprobs = student_probs_full[row_indices, id_tensor].log() # student_probs_full[row_indices][id_tensor] -> corresponding probs
                        student_logprobs = student_logprobs.reshape(-1)

                        loss = F.kl_div(student_logprobs, teacher_probs_tensor, reduction="sum") / max_tokens
                        if (batch_loss == 0):
                            batch_loss = loss
                        else:
                            batch_loss = batch_loss + loss

                    logger.info(f"loss = {batch_loss}")

                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += batch_loss.detach().float()
                    accelerator.backward(batch_loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()


                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1


            # Evaluation
            student_model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                
                with torch.no_grad():
                    prompt = batch['text'][0]
                    eval_batch = tokenizer(prompt, return_tensors='pt').to("cuda")
                    outputs = student_model(**eval_batch)

            losses = torch.tensor(0, dtype=torch.float) # loss to 0
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    # Save checkpoint
    student_model.save_pretrained('student_model')

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(student_model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()