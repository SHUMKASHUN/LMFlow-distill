import logging
import sys
import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
import TeacherDataset
import argparse
import math
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
import wandb

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
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="train batch size per device") #8
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="eval batch size per device") #8
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", type=int, default=1e10, help="max steps for debug.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
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

    # setup logging
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

    accelerator = Accelerator()
    device = accelerator.device

    # setup student model
    logger.info("*** [START] Setting up student model ***")
    config = AutoConfig.from_pretrained(student_name)
    tokenizer = AutoTokenizer.from_pretrained(student_name, use_fast=False)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        from_tf=False,
        torch_dtype=torch.bfloat16
        # config=config,
    )
    student_model.to(device)
    logger.info("*** [FINISH] Setting up student model ***")


    # dataloader
    logger.info("*** [START] Creating dataloader ***")

    data_path = './dataset/Robin/0-120.jsonl'
    # data_path = '/home/ksshumab/DistillData/LMFlow/distilled_data.jsonl'
    teacher_dataset = TeacherDataset(data_path)
    train_dataloader = DataLoader(teacher_dataset, 
                                  batch_size=args.per_device_train_batch_size, 
                                  collate_fn=teacher_dataset.collate_fn)
    eval_dataloader = train_dataloader # for debug only
    logger.info("*** [FINISH] Creating dataloader ***")


    # optimizer
    logger.info("*** [START] Setting up optimizer and scheduler ***")
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # dummy optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim 
    )
    optimizer = optimizer_cls(student_model.parameters(), lr=args.learning_rate)
    # dummy scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )
    # prepare with accelerator
    student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # recalculate training steps due to multi-cpus
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    logger.info("*** [FINISH] Setting up optimizer and scheduler ***")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # log training info
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    # wandb setup
    if accelerator.is_local_main_process:
        wandb.init(
            project = "distill-llama-7b",
            config = {
                "data_path": './dataset/Robin/0-120.jsonl',
                "batch_size": args.per_device_train_batch_size,
                "epoch": args.num_train_epochs,
            }
        )

    # distill from soft probs
    for epoch in range(starting_epoch, args.num_train_epochs):
        print("epoch", epoch)
        student_model.train()
        if args.with_tracking:
            total_loss = 0

        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                with accelerator.autocast():
                    
                    # extract info from batch 
                    input_token = batch['input_token'] # [b, 512] list
                    output_token = batch['output_token'] # [b, 511, 5] list
                    top_logprob = batch['top_logprob'] # [b, 511, 5] list
                    # to tensor
                    input_tokens = torch.Tensor(input_token).to(torch.int32).to(device)
                    output_tokens = torch.Tensor(output_token).to(torch.int64).to(device)
                    teacher_top_logprob = torch.Tensor(top_logprob).to(device)

                    # student output
                    student_outputs = student_model(input_tokens)
                    student_logits = student_outputs.logits # student_outputs.logits.shape = torch.Size([2,512,32000])
                    student_prob = F.softmax(student_logits/student_temp, dim=-1) # [2,512,32000]

                    # search for student top prob by teacher output token
                    student_top_prob = torch.gather(student_prob[:,1:,:],-1,output_tokens) # [2,511,5]

                    # process
                    teacher_top_prob = teacher_top_logprob.exp() # convert to regular prob
                    student_logsoftmax = F.log_softmax(student_top_prob/student_temp, dim=-1)
                    teacher_softmax = F.softmax(teacher_top_prob/student_temp, dim=-1)

                    # kl div
                    batch_loss = F.kl_div(student_logsoftmax, teacher_softmax, reduction="batchmean")
                    logger.info(f"loss = {batch_loss}")

                    logger.info(f"loss = {batch_loss}")
                    if accelerator.is_local_main_process:
                        wandb.log({"loss": batch_loss})

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

    # Save checkpoint
    logger.info("*** [START] Saving Pre-trained Model ***")
    student_model.save_pretrained('student_model')
    logger.info("*** [FINISH] Finish Saving Pre-trained Model ***")
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