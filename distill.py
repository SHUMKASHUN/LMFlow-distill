import argparse
import json
import logging
import math
import os
import sys
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
from accelerate.logging import get_logger
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
from TeacherDataset import TeacherDataset

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM-Distill")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="train batch size per device")
    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="Where to store the final model.")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--beta_1", type=float, default=0.9, help="AdamW Optimizer Beta 1")
    parser.add_argument("--beta_2", type=float, default=0.999, help="AdamW Optimizer Beta 2")
    parser.add_argument("--eps", type=float, default=1e-6, help="AdamW Optimizer epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW Optimizer weight decay")
    parser.add_argument("--lr_scheduler_type",
                        type=SchedulerType,
                        default="cosine",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        )
    parser.add_argument("--gradient_checkpointing", default=False, action='store_true', help="Whether to enable gradient checkpointing")
    parser.add_argument("--teacher_name", type=str, default="robin-33b", choices=["robin-33b"], help="teacher model name")
    parser.add_argument("--student_name", type=str, default="pinkmanlove/llama-7b-hf", help="student model name") 
    parser.add_argument("--dataset_name_or_path", type=str, 
                        default="./datasets/Train/33b_blocksize_512_v2.jsonl",
                        help="dataset name or path")
    parser.add_argument("--percentage", type=float, default=1.0, help="Percentage that partition dataset.")
    parser.add_argument("--wandb_name", type=str, default="distill_llama7b", help="The wandb visulization name.")

    parser.add_argument("--max_tokens", type=int, default=3, help="minimum length for generation")
    parser.add_argument("--max_num_log_probs", type=int, default=5, help="minimum length for generation")
    parser.add_argument("--stop_token", default=None, help="stop token of GPT-3, choice=[\n, None],")
    parser.add_argument("--teacher_temp", type=float, default=1.0, help="temperature of the teacher")
    parser.add_argument("--student_temp", type=float, default=1.0, help="temperature of the student")
    parser.add_argument("--log_level", type=str, default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="logging level")
    parser.add_argument("--local_rank", type=int, help="local rank")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--max_steps", type=int, default=1e10, help="max steps for debug.")
    parser.add_argument("--method", type=str, default="forward_kl_text_only")
    parser.add_argument("--use_other_token", type=bool, default=False)
    parser.add_argument("--use_mixed_training", type=bool, default=False)

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
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    student_name = args.student_name
    student_temp = args.student_temp
    set_seed(args.random_seed)

    # Accelerator setup 
    accelerator = Accelerator()
    device = accelerator.device

    # wandb setup
    if accelerator.is_local_main_process:
        wandb.init(
            project = "distill-llama-7b",
            group = "llama-7b",
            name = args.wandb_name,
            config = {
                "model": {"teacher": args.teacher_name, "student": args.student_name},
                "data path": args.dataset_name_or_path,
                "parameters": {"epoch": args.num_train_epochs, "batch size": args.per_device_train_batch_size},
                "optimizer": {"learning rate": args.learning_rate, "beta_1": args.beta_1, "beta_2": args.beta_2, "epsilon": args.eps},
                "scheduler": args.lr_scheduler_type
            }
        )

    # logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = get_logger(__name__)
    logger.info(f"Arguments : {args}")

    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    # setup student model
    logger.info("*** [START] Setting up student model ***")
    tokenizer = AutoTokenizer.from_pretrained(student_name, use_fast=False)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        from_tf=False,
        torch_dtype=torch.bfloat16
    )
    if (args.gradient_checkpointing == True):
        student_model.gradient_checkpointing_enable()
    student_model.to(device)
    logger.info("*** [FINISH] Setting up student model ***")


    # dataloader
    logger.info("*** [START] Creating dataloader ***")
    teacher_dataset = TeacherDataset(args.dataset_name_or_path, args.percentage)
    train_dataloader = DataLoader(teacher_dataset, 
                                  batch_size=args.per_device_train_batch_size, 
                                  collate_fn=teacher_dataset.collate_fn)
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
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.beta_1, args.beta_2),
        "eps": args.eps,
    } 
    optimizer = optimizer_cls(student_model.parameters(), **optimizer_kwargs)
    # dummy scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=0.03 * args.max_train_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )
    # prepare with accelerator
    student_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        student_model, optimizer, train_dataloader, lr_scheduler
    )

    # recalculate training steps due to multi-gpus
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

    # distill from soft probs
    for epoch in range(starting_epoch, args.num_train_epochs):
        student_model.train()
        if args.with_tracking:
            total_loss = 0
        step = 0
        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                with accelerator.autocast():
                    
                    # extract info from batch 
                    input_token = torch.Tensor(batch['input_token']).to(torch.int32).to(device) # [b, 512]
                    output_token = torch.Tensor(batch['output_token']).to(torch.int64).to(device) # [b, 512, 5]
                    teacher_top_prob = torch.Tensor(batch['top_prob']).to(device) # [b, 512, 5]
                    attention_mask = torch.Tensor(batch['attention_mask']).to(torch.int8).to(device) # [b, 512]
                    loss_mask = torch.Tensor(batch['loss_mask']).to(torch.int8).to(device) # [b, 512]

                    # get student output
                    student_outputs = student_model(input_ids=input_token, attention_mask=attention_mask)
                    student_logits = student_outputs.logits # [b, 512, 32000]
                    student_prob = F.softmax(student_logits/student_temp, dim=-1) # [b, 512, 32000]

                    # select student top prob by teacher output token
                    student_top_prob = torch.gather(student_prob, -1, output_token) # [b, 512, 5]

                    # do not use other token; take log-softmax directly
                    if(args.use_other_token == False):
                        student_logsoftmax = F.log_softmax(student_top_prob/student_temp, dim=-1)
                        teacher_logsoftmax = F.log_softmax(teacher_top_prob/student_temp, dim=-1)
                    # use other token; take log only
                    else:
                        sum_student_top_prob = torch.sum(student_top_prob, dim=-1) # [b, 512]
                        sum_teacher_top_prob = torch.sum(teacher_top_prob, dim=-1) # [b, 512]
                        one_prob = torch.ones(sum_student_top_prob.shape).to(device) # [b, 512]
                        student_ot_prob = torch.sub(one_prob, sum_student_top_prob).unsqueeze(-1) # [b, 512, 1]
                        teacher_ot_prob = torch.sub(one_prob, sum_teacher_top_prob).unsqueeze(-1) # [b, 512, 1]
                        
                        student_prob = torch.cat((student_top_prob, student_ot_prob), dim=-1) # [b, 512, 6]
                        teacher_prob = torch.cat((teacher_top_prob, teacher_ot_prob), dim=-1) # [b, 512, 6]
                        # student_logsoftmax = torch.log(student_prob)
                        # teacher_logsoftmax = torch.log(teacher_prob)
                        student_logsoftmax = F.log_softmax(student_prob/student_temp, dim=-1)
                        teacher_logsoftmax = F.log_softmax(teacher_prob/student_temp, dim=-1)

                    # calculate loss
                    batch_loss = 0
                    ## forward kl div
                    if(args.method == "forward_kl_text_only"): 
                        batch_loss = F.kl_div(student_logsoftmax, teacher_logsoftmax, reduction="batchmean", log_target=True)
                    ## reverse kl div
                    elif(args.method == "reverse_kl_text_only"): 
                        batch_loss = F.kl_div(teacher_logsoftmax, student_logsoftmax, reduction="batchmean", log_target=True)
                    ## absolute kl div
                    elif(args.method == "abs_kl_text_only"):
                        kl_result = F.kl_div(teacher_logsoftmax, student_logsoftmax, reduction="none", log_target=True)
                        batch_loss = torch.sum(torch.abs(kl_result))
                    ## apply musk
                    elif(args.method == "forward_kl_text2text"):
                        for i in range(args.per_device_train_batch_size): # batch
                            s = student_logsoftmax[i][loss_mask[i] == 1] # [512, 5]
                            t = teacher_logsoftmax[i][loss_mask[i] == 1]
                            batch_loss = batch_loss + F.kl_div(s, t, reduction="sum", log_target=True)
                        batch_loss = batch_loss/args.per_device_train_batch_size
                    elif(args.method == "reverse_kl_text2text"):
                        for i in range(args.per_device_train_batch_size): # batch
                            s = student_logsoftmax[i][loss_mask[i] == 1]
                            t = teacher_logsoftmax[i][loss_mask[i] == 1]
                            batch_loss = batch_loss + F.kl_div(t, s, reduction="sum", log_target=True)
                        batch_loss = batch_loss/args.per_device_train_batch_size
                    elif(args.method == "abs_kl_text2text"):
                        for i in range(args.per_device_train_batch_size): # batch
                            s = student_logsoftmax[i][loss_mask[i] == 1]
                            t = teacher_logsoftmax[i][loss_mask[i] == 1]
                            kl_result = F.kl_div(t, s, reduction="none", log_target=True)
                            batch_loss = batch_loss + torch.sum(torch.abs(kl_result))
                        batch_loss = batch_loss/args.per_device_train_batch_size
                    else:
                        raise NotImplementedError(f"{args.method} not implemented")

                    # log info
                    try:
                        last_lr = lr_scheduler.get_last_lr()[0]
                        if torch.is_tensor(last_lr):
                            last_lr = last_lr.item()
                    except:
                        last_lr = 0
                    logger.info(f"STEP : {step} / {args.max_train_steps}, loss = {batch_loss}, learning rate = {last_lr} ")
                    if accelerator.is_local_main_process:
                        wandb.log({"loss": batch_loss, "learning_rate": last_lr})
                    if args.with_tracking:
                        total_loss += batch_loss.detach().float()
                    accelerator.backward(batch_loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    step = step + 1

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

        # Save checkpoint
        logger.info("*** [START] Saving Pre-trained Model ***")
        if args.with_tracking:
            accelerator.end_training()

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.save_pretrained(
                args.output_dir + f"/epoch_{epoch}/",
                is_main_process=accelerator.is_main_process, 
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(student_model),
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir + f"/epoch_{epoch}/")
        logger.info("*** [FINISH] Finish Saving Pre-trained Model ***")

if __name__ == "__main__":
    main()



