import openai
import logging
import sys
import random
import torch
import os
import json
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
from torch.utils.data import DataLoader
import math
import csv
from accelerate import Accelerator

API_KEY_LIST = ["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",
                "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",
                "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",
                "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",
                "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"]
NUM_API_KEYS = len(API_KEY_LIST)

def complete_chatgpt(chat):
    response = None
    received = False

    key_index = random.randint(0, NUM_API_KEYS - 1)
    openai.api_key = API_KEY_LIST[key_index]

    while not received:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=chat
            )
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{chat}\n\n")
                assert False
            print("API error:", sys.exc_info())

            key_index = random.randint(0, NUM_API_KEYS - 1)
            openai.api_key = API_KEY_LIST[key_index]
            print("openai.api_key", openai.api_key)
    return response["choices"][0]["message"]["content"]

def complete_gpt3(prompt, max_tokens, model_name, temp=0.0, num_log_probs=None, echo=False, stop=None):
    # call GPT-3 code-davinci and text-davinci API until result is provided and then return it
    response = None
    received = False

    key_index = random.randint(0, NUM_API_KEYS - 1)
    openai.api_key = API_KEY_LIST[key_index]

    while not received:
        try:
            response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=max_tokens, temperature=temp,
                                                logprobs=num_log_probs, echo=echo, stop=stop)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", sys.exc_info())

            key_index = random.randint(0, NUM_API_KEYS - 1)
            openai.api_key = API_KEY_LIST[key_index]
            print("openai.api_key", openai.api_key)
    return response

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
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="learning rate")
    parser.add_argument("--dataset_name", type=str, default="chatgpt-prompt", help="dataset name")
    parser.add_argument("--teacher_name", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "code-davinci-002", "text-davinci-002"], help="teacher model name")
    parser.add_argument("--student_name", type=str, default="gpt2", help="student model name")
    parser.add_argument("--max_tokens", type=int, default=3, help="minimum length for generation")
    parser.add_argument("--max_num_log_probs", type=int, default=5, help="minimum length for generation")
    parser.add_argument("--stop_token", default=None, help="stop token of GPT-3, choice=[\n, None],")
    parser.add_argument("--teacher_temp", type=float, default=0.7, help="temperature of the teacher")
    parser.add_argument("--student_temp", type=float, default=0.7, help="temperature of the student")
    parser.add_argument("--log_level", type=str, default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="logging level")
    parser.add_argument("--validation_split_percentage", type=int, default=20, help="the percentage of validation split")
    parser.add_argument("--demo_example_in_prompt", type=bool, default=False, help="When this flag is True, the prompt will include examplary, samples in the prompt if available from the dataset.")
    parser.add_argument("--local_rank", type=int, help="local rank")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="train batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="eval batch size per device")
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
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # set up student model
    logger.info("*** [START] Setting up student model ***")
    config = AutoConfig.from_pretrained(student_name)
    tokenizer = AutoTokenizer.from_pretrained(student_name, use_fast=True)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        from_tf=False,
        config=config,
    )
    logger.info("*** [FINISH] Setting up student model ***")

    # load dataset
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
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size
    ) # , collate_fn=default_data_collator, 
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.per_device_eval_batch_size
    )
    logger.info("*** [FINISH] Creating dataloader ***")

    logger.info("*** [START] Setting up optimizer and scheduler ***")
    # Optimizer
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

    if args.teacher_name == "gpt-3.5-turbo":
        # distill from hard tokens.
        for epoch in range(starting_epoch, args.num_train_epochs):
            student_model.train()
            if args.with_tracking:
                total_loss = 0
            all_history = []
            for step, batch in enumerate(train_dataloader):
                if step > args.max_steps: break
                prompt = batch['text'][0] # seems that chatgpt can only deal with one input instead of batch processing
                with accelerator.accumulate(student_model):
                    history = [{"role": "system", "content": f"You are a helpful assistant."}]
                    history.append({"role": "user", "content": prompt})
                    # print("prompt", prompt)
                    teacher_generation = complete_chatgpt(history)
                    history.append({"role": "assistant", "content": teacher_generation})
                    all_history.append(history)
                    full_conversation = "user: " + prompt + "assistant: " + teacher_generation

                    # print("full_conversation", full_conversation)
                    teacher_batch = tokenizer(full_conversation, return_tensors='pt', truncation=True, max_length=max_tokens).to("cuda")
                    # teacher_batch["input_ids"] = teacher_batch["input_ids"][:, :max_tokens] # limit the length
                    # teacher_batch["attention_mask"] = teacher_batch["attention_mask"][:, :max_tokens] # limit the length
                    teacher_batch["labels"] = teacher_batch["input_ids"].detach().clone()
                    outputs = student_model(**teacher_batch)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.float()
                        accelerator.log(
                            {
                                "train_loss_step": loss.float(),
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
            
            student_model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    prompt = batch['text'][0]
                    history = [{"role": "system", "content": f"You are a helpful assistant."}]
                    history.append({"role": "user", "content": prompt})
                    teacher_generation = complete_chatgpt(history)
                    history.append({"role": "assistant", "content": teacher_generation})
                    all_history.append(history)
                    full_conversation = "user: " + prompt + "assistant: " + teacher_generation
                    teacher_batch = tokenizer(full_conversation, return_tensors='pt').to("cuda")
                    teacher_batch["labels"] = teacher_batch["input_ids"].detach().clone()

                    outputs = student_model(**teacher_batch)
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
            
            save_history(all_history, args.output_dir)
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
        
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")


    elif args.teacher_name in ["code-davinci-002", "text-davinci-002"]: 
        # distill from soft probs.
        for epoch in range(starting_epoch, args.num_train_epochs):
            print("epoch", epoch)
            student_model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                prompt = batch['text']

                with accelerator.accumulate(student_model):
                    teacher_generation = complete_gpt3(prompt, max_tokens=max_tokens, model_name=teacher_name, num_log_probs=max_num_log_probs, echo=True, stop=stop, temp=teacher_temp)
                    teacher_top_logprobs = teacher_generation["choices"][0]["logprobs"]["top_logprobs"]
                    teacher_output_tokens = teacher_generation["choices"][0]["text"]

                    teacher_logprobs_list = []
                    id_list = []
                    for teacher_step in teacher_top_logprobs[-max_tokens:]:
                        teacher_logprobs_list.append([])
                        id_list.append([])
                        for token, logprob in teacher_step.items():
                            id = tokenizer.encode(token)[0]
                            teacher_logprobs_list[-1].append(logprob)
                            id_list[-1].append(id)

                    teacher_logprobs_tensor = torch.tensor(teacher_logprobs_list).reshape(-1).to("cuda")
                    teacher_probs_tensor = teacher_logprobs_tensor.exp()
                    id_tensor = torch.tensor(id_list)

                    # student
                    teacher_batch = tokenizer(teacher_output_tokens, return_tensors='pt').to("cuda")
                    # teacher_batch = accelerator.prepare(teacher_batch)

                    outputs = student_model(**teacher_batch) # student_model: cuda
                    student_logits = outputs.logits[0][-max_tokens:]
                    student_probs_full = F.softmax(student_logits / student_temp, dim=1)
                    
                    row_indices = torch.tensor([[i] * max_num_log_probs for i in range(max_tokens)])
                    row_indices = row_indices.reshape(-1)
                    id_tensor = id_tensor.reshape(-1)

                    student_logprobs = student_probs_full[row_indices, id_tensor].log()
                    student_logprobs = student_logprobs.reshape(-1)

                    loss = F.kl_div(student_logprobs, teacher_probs_tensor, reduction="sum") / max_tokens
                    logger.info(f"loss = {loss}")

                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

            student_model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = student_model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

            losses = torch.cat(losses)
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
            # if args.push_to_hub:
            #     repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            # with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            #     json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()