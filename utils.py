# This file contains necessary helper functions
# e.g. GPT request, create_dataloader
import openai # comment?
import random
import sys
import numpy as np
import torch
import argparse
from pathlib import Path
from filelock import Timeout, FileLock
import json
import jsonlines
import logging
import re
from collections import Counter
import time
import git
import os
import shutil
import pandas as pd

from datasets import concatenate_datasets, load_dataset, load_from_disk
from datasets import Dataset, DatasetDict
from transformers.trainer_utils import is_main_process
from transformers import AutoTokenizer
# from API_POOL_REPO import *

logger = logging.getLogger(__name__)

class DownloadError(Exception):
    pass

# put your API key in the list
# NO_SOLUTION = '<NO_SOL>'
NO_SOLUTION = '-10086'

# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# pass in a list of prompts and returns a response body contains a list of responses
def GPT3_request(model:str, input_prompt:list, max_tokens:int, temperature=0.7, stop=None, worker_id=None, API_PARTITION_POOL=None):
    resp = None
    done = False
    while not done:
        try:
            # random select key
            # key_index = random.randint(0, NUM_API_KEYS - 1)
            # openai.api_key = API_KEY_POOL[key_index]

            # api key polling request
            key_index = API_PARTITION_POOL[worker_id]['cur_index']
            openai.api_key = API_PARTITION_POOL[worker_id]["keys"][key_index]
            resp = openai.Completion.create(
                model=model,
                prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
            key_index += 1
            if key_index == len(API_PARTITION_POOL[worker_id]['keys']):
                API_PARTITION_POOL[worker_id]['cur_index'] = 0
            else:
                API_PARTITION_POOL[worker_id]['cur_index'] = key_index
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt}\n")
                print(f"Reason: {errno[1]}")
                key_index = API_PARTITION_POOL[worker_id]['cur_index']
                print(f"invalid key: {API_PARTITION_POOL[worker_id]['keys'][key_index]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")

            key_index = API_PARTITION_POOL[worker_id]['cur_index']
            key_index += 1
            if key_index == len(API_PARTITION_POOL[worker_id]['keys']):
                API_PARTITION_POOL[worker_id]['cur_index'] = 0
            else:
                API_PARTITION_POOL[worker_id]['cur_index'] = key_index
            time.sleep(3)
    return resp


def load_data(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()
    # Use tokenizer to implement truncation
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))
    elif args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                qes = json_res["question"].strip() + " Answer Choices:"

                for opt in json_res["options"]:
                    qes += f" ({opt}"

                questions.append(qes)
                answers.append(json_res["correct"])
    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "asdiv":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["Instances"]
            for line in json_data:
                q = line['input'].strip()
                a = line['output'][0]
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("addsub", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "csqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    elif args.dataset == "strategyqa":
        if 'task' in args.dataset_path:
            with open(args.dataset_path) as f:
                json_data = json.load(f)["examples"]
                for line in json_data:
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    questions.append(q)
                    answers.append(a)
        else:
            with open(args.dataset_path, encoding='utf-8') as f:
                json_data = json.load(f)
                for line in json_data:
                    q = line["question"].strip() 
                    if line['answer']:
                        a = 'yes'
                    else:
                        a = 'no'
                    questions.append(q)
                    answers.append(a)
    elif args.dataset in ("coin_flip", "last_letters", "chinese_test", "example_dataset"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "pubmedqa":
        with open(args.dataset_path, encoding='utf-8') as f:
            json_data = json.load(f)
            for line in json_data:
                tokens = tokenizer.tokenize(line["context"])
                if(len(tokens) > 850):
                    truncated = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[0:850]))
                    q = "Context: " + truncated + " Question: " +  line["question"].strip() + " yes or no?"
                else:
                    q = "Context: " + line["context"] + " Question: " +  line["question"].strip() + " yes or no?"
                questions.append(q)
                answers.append(line["final_decision"])
    elif args.dataset == "medmcqa":
        with open(args.dataset_path, encoding='utf-8') as f:
            json_data = json.load(f)
            for line in json_data:
                q = "Question: " + line["question"] + " (A) " + line["opa"] + \
                        " (B) " + line["opb"] +  " (C) " + line["opc"] +  " (D) " + line["opd"] + "."
                questions.append(q)
                answers.append(line["answer"])  
    elif args.dataset == "usmle":
        with open(args.dataset_path, encoding='utf-8') as f:
            json_data = json.load(f)
            for line in json_data:
                q = "Question: " + line["question"] + " (A) " + line["opa"] + \
                        " (B) " + line["opb"] +  " (C) " + line["opc"] +  " (D) " + line["opd"] + "."
                questions.append(q)
                answers.append(line["answer"].lower())  
    else:
        raise NotImplementedError

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    datasize = len(questions)
    return questions, answers, datasize


def download_natural_inst_dataset(local_rank):
    project_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(project_dir, "tmp/dataset")
    data_dir = os.path.join(project_dir, "dataset/natural-instructions")
    lock_path = os.path.join(tmp_dir, "natural_inst_download.lock")
    complete_mark = os.path.join(tmp_dir, "natural_inst_download_complete.mark")

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path)

    class Progress(git.remote.RemoteProgress):
        def update(self, *args):
            print(self._cur_line)

    # In case multiple processes trigger the downloading simultaneously,
    # the first process handles the download
    if is_main_process(local_rank):
        with lock:
            if not os.path.exists(complete_mark):
                try:
                    if os.path.exists(data_dir):
                        shutil.rmtree(data_dir)
                    Path(data_dir).mkdir(parents=True, exist_ok=True)
                    print("Downloading Super Natural Instructions"
                          "Dataset from its official git repository...")
                    repo = git.Repo.clone_from(
                        "https://github.com/allenai/natural-instructions.git",
                        data_dir,
                        progress=Progress()
                    )
                    repo.git.checkout("v2.8")
                    Path(complete_mark).touch()
                except BaseException:
                    shutil.rmtree(data_dir)
                    raise DownloadError("Downloading process is interrupted.")
    else:
        # Waits until main process finishes downloading
        print("Waiting for other processes to download the dataset...")
        while not os.path.exists(complete_mark):
            time.sleep(10)

    task_dir = os.path.join(data_dir, "tasks")
    task_file_list = [ x.absolute().as_posix()
                       for x in Path(task_dir).glob("*.json") ]
    return task_file_list


def get_prompt(definition, input_text, output_text, demo_example_in_prompt, pos_example_input, pos_example_output, prompt_structure):
    if not demo_example_in_prompt:
        prompt = prompt_structure.format(
            definition=definition,
            input=input_text,
            output=output_text,
        )
    else:
        prompt = prompt_structure.format(
            definition=definition,
            input=input_text,
            output=output_text,
            example_input=pos_example_input,
            example_output=pos_example_output,
        )
    return prompt


def get_dataset(split, pos_example_input, pos_example_output, definition, extensions, task_file, dataset_args, demo_example_in_prompt, preprocessing_num_workers, prompt_structure):
    raw_dataset = load_dataset(
        extensions,
        data_files=task_file,
        field="Instances",
        split=split,
        use_auth_token=None,
        **dataset_args,
    )

    def get_samples(examples):
        return {
            "text": [ get_prompt(definition, x, y, demo_example_in_prompt, pos_example_input, pos_example_output, prompt_structure)
                      for (x, y_list) in zip(examples["input"], examples["output"])
                          for y in y_list
        ]}

    task_dataset = raw_dataset.map(
        get_samples,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=raw_dataset.features,
        desc=f"Getting prompt samples from raw dataset.",
    )
    return task_dataset

def load_customized_dataset(
    task_file_list: list,
    validation_split_percentage: float,
    demo_example_in_prompt: bool = False,
    preprocessing_num_workers: int = 1,
    prompt_structure: str = None,
): 
    if validation_split_percentage is None:
        validation_split_percentage = 0

    train_dataset_list = []
    validation_dataset_list = []
    # for task_file in task_file_list:
    for index, task_file in enumerate(task_file_list):
        if(index == 1):
            break
        dataset_args = {}
        extensions = 'json'

        with open(task_file) as fin:
            json_data = json.load(fin)
            definition = ''.join(json_data["Definition"])
            pos_example_input = json_data["Positive Examples"][0]["input"]
            pos_example_output = json_data["Positive Examples"][0]["output"]
            pos_example_explan = json_data["Positive Examples"][0]["explanation"]
            neg_example_input = json_data["Negative Examples"][0]["input"]
            neg_example_explan = json_data["Negative Examples"][0]["explanation"]

        train_dataset = get_dataset(split=f"train[{validation_split_percentage}%:]",
                                    pos_example_input=pos_example_input,
                                    pos_example_output=pos_example_output,
                                    definition=definition,
                                    extensions=extensions,
                                    task_file=task_file,
                                    dataset_args=dataset_args,
                                    demo_example_in_prompt=demo_example_in_prompt,
                                    preprocessing_num_workers=preprocessing_num_workers,
                                    prompt_structure=prompt_structure)

        validation_dataset = get_dataset(split=f"train[:{validation_split_percentage}%]",
                                    pos_example_input=pos_example_input,
                                    pos_example_output=pos_example_output,
                                    definition=definition,
                                    extensions=extensions,
                                    task_file=task_file,
                                    dataset_args=dataset_args,
                                    demo_example_in_prompt=demo_example_in_prompt,
                                    preprocessing_num_workers=preprocessing_num_workers,
                                    prompt_structure=prompt_structure)

        train_dataset_list.append(train_dataset)
        validation_dataset_list.append(validation_dataset)

    train_datasets = concatenate_datasets(train_dataset_list)
    validation_datasets = concatenate_datasets(validation_dataset_list)
    if len(validation_datasets) == 0:
        validation_datasets = validation_datasets.add_column(name="text", column=[])
    raw_datasets = DatasetDict({
        "train": train_datasets,
        "validation": validation_datasets
    })
    return raw_datasets


def load_special_dataset_for_train(
    dataset_name: str,
    validation_split_percentage: float,
    demo_example_in_prompt: bool = False,
    local_rank: int = -1,
    preprocessing_num_workers: int = 1,
    cache_dir: str = ".cache/llm-ft/datasets",
    # prompt_structure: str = None,
    prompt_structure: str = "{input}",
):
    # Customized cache mechanism is necessary, since some customized datasets,
    # e.g. super-natural-instructions, have a lot of small datasets, where
    # huggingface's original cache mechanism doesn't work well (still slow).
    # By default the cache_path is "~/{cache_dir}/{dataset_name}"

    project_dir = os.path.dirname(__file__)
    data_dir = os.path.join(project_dir, f"dataset/{dataset_name}")
    if not os.path.exists(data_dir):
        data_dir = dataset_name

    if dataset_name == "super-natural-instructions":
        task_file_list = download_natural_inst_dataset(local_rank)
    else:
        task_file_list = [ x.absolute().as_posix()
                       for x in Path(data_dir).glob("*.json") ]
    raw_datasets = load_customized_dataset(
        task_file_list,
        validation_split_percentage,
        demo_example_in_prompt,
        preprocessing_num_workers=preprocessing_num_workers,
        prompt_structure=prompt_structure,
    )

    return raw_datasets


# process the dataset into a loader of batches
def batchlize(examples:list, batch_size:int, random_shuffle:bool):
    size = 0
    questions = []
    length = len(examples)
    if (random_shuffle):
        random.shuffle(examples)
    while size < length:
        if length - size > batch_size:
            questions.append(examples[size:size+batch_size])
            size += batch_size
        else:
            questions.append(examples[size:size+(length-size)])
            size += (length - size)
    return questions


# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(args)->list:
    set_random_seed(args.random_seed)
    questions, answers, datasize = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    dataloader = batchlize(dataset, args.minibatch_size, args.random_shuffle)
    print(f"dataloader size: {len(dataloader)}")
    return dataloader, datasize


# read the generated/prepared prompt json file
# return a string of prefix prompt before each question
def create_input_prompt(args, cot_flag:bool)->str:
    x, z, y = [], [], []
    
    with open(args.prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["prompt"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))
    
    prompt_text = ""
    for i in index_list:
        if cot_flag:
            if args.dataset == "strategyqa":
                prompt_text += x[i] + " " + z[i] + " " + \
                            "So the answer is" + " " + y[i] + ".\n\n"
            else:
                prompt_text += x[i] + " " + z[i] + " " + \
                            args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return prompt_text

def answer_extraction(args, response):   #use this funtion to extract answers from generated text
    # temp = response["generated_text"]
    temp = response
    if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua", "csqa"):
        temp = re.findall(r'A|B|C|D|E', temp)
    elif args.dataset in ("strategyqa", "coin_flip"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif args.dataset in ("last_letters"):
        temp = re.sub("\"|\'|\n|\.|\s","", temp)
        temp = [temp]
    elif args.dataset == "pubmedqa":
        # pattern = "Output: (yes|no|maybe)"
        # sttr = re.search(pattern, temp)
        # answer = sttr.group(0)[8:] if sttr is not None else "N/A"
        pattern = "(answer|Answer|ANSWER|output|Output|OUTPUT|A): \(*(yes|Yes|YES|no|No|NO|maybe|Maybe|MAYBE)"
        sttr = re.search(pattern, temp)
        if sttr is not None:
            mid_answer = sttr.group(0)
            mid_answer = mid_answer.split(":")[-1].strip()
            answer = mid_answer.lower()
        else:
            pattern = "(yes|Yes|YES|no|No|NO|maybe|Maybe|MAYBE)(\.|\s)"
            sttr = re.search(pattern, temp)
            if sttr is not None:
                answer = sttr.group(0)[:-1].lower()
            else:
                answer = "N/A"
        return answer
    elif args.dataset == "medmcqa":
        # pattern = "Output: (A|B|C|D)."
        # sttr = re.search(pattern, temp)
        # answer = sttr.group(0)[8:-1].lower() if sttr is not None else "N/A"
        pattern = "(answer|Answer|ANSWER|output|Output|OUTPUT|A): \(*(A|B|C|D|a|b|c|d)"
        sttr = re.search(pattern, temp)
        if sttr is not None:
            mid_answer = sttr.group(0)
            answer = mid_answer[-1].lower()
        else:
            pattern = "(A|B|C|D|a|b|c|d)(\.|\s)"
            sttr = re.search(pattern, temp)
            if sttr is not None:
                answer = sttr.group(0)[0].lower()
            else:
                answer = "N/A"
        return answer

    elif args.dataset == "usmle":
        # pattern = "Output: (A|B|C|D)."
        # sttr = re.search(pattern, temp)
        # answer = sttr.group(0)[8:-1].lower() if sttr is not None else "N/A"
        pattern = "(Answer|Output|A): \(*(A|B|C|D|a|b|c|d)"
        sttr = re.search(pattern, temp)
        if sttr is not None:
            mid_answer = sttr.group(0)
            answer = mid_answer[-1].lower()
        else:
            pattern = "(A|B|C|D|a|b|c|d)(\.|\s)"
            sttr = re.search(pattern, temp)
            if sttr is not None:
                answer = sttr.group(0)[0].lower()
            else:
                answer = "N/A"
        return answer

    if len(temp) != 0:
        answer = temp[-1]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

            # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif args.dataset in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""
    else:
        answer = ""
    return answer

def find_most_frequent(arr, n):
    # method 1: return max(arr[:n], key=arr.count)
    # method 2:
    arr_acounts = Counter(arr[:n])
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]
    return frequency, most_frequent_item
