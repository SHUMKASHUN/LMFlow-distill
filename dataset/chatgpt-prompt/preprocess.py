from datasets import load_dataset
import json

chatgpt_dataset1 = load_dataset("fka/awesome-chatgpt-prompts")["train"]

print(chatgpt_dataset1)

def generate_dataset1():
    return_chatgpt_dataset= []
    for index in range(0,len(chatgpt_dataset1)):
        temp = {}
        temp["act"] = chatgpt_dataset1[index]["act"]
        temp["prompt"] =  chatgpt_dataset1[index]["prompt"]
        return_chatgpt_dataset.append(temp)
    return return_chatgpt_dataset

def reformat_for_lm_training():
    return_json_pqa_lm = {}
    list_pqa_lm = []
    for index in range(0,len(chatgpt_dataset1)):
        temp = {}
        temp["id"] = index
        temp["input"] = chatgpt_dataset1[index]["prompt"]
        temp["output"] = [""]
    
        list_pqa_lm.append(temp)
    
    return_json_pqa_lm["Contributors"] = "KaShun SHUM","Shizhe Diao",
    return_json_pqa_lm["Source"] = "chatgpt prompt",
    return_json_pqa_lm["URL"] = "https://huggingface.co/datasets/NA",
    return_json_pqa_lm["Categories"] = "Question Answering",
    return_json_pqa_lm["Reasoning"] = "Multihop Reasoning",
    return_json_pqa_lm["Definition"] = "",
    return_json_pqa_lm["Input_language"] = "English", 
    return_json_pqa_lm["Output_language"] = "English",
    return_json_pqa_lm["Instruction_language"] = "English",  
    return_json_pqa_lm["Domains"] = "Medical",    
    return_json_pqa_lm["Positive Examples"] =  { "input": "", "output": "",  "explanation": ""} , 
    return_json_pqa_lm["Negative Examples"] =  { "input": "", "output": "",  "explanation": ""} ,
    return_json_pqa_lm["Instances"] =  list_pqa_lm
    
    return return_json_pqa_lm

if __name__ == '__main__':

    # return_list = generate_dataset1()
    # with open("./chatgpt_prompt_{}.json".format(len(return_list)),"w") as f:
    #     json.dump(return_list,f,indent=2)   
    
    return_list_pqa_lm = reformat_for_lm_training()
    with open("./chatgpt_prompt_for_lm_{}.json".format(len(return_list_pqa_lm["Instances"])),"w") as f:
        json.dump(return_list_pqa_lm,f,indent=4) 