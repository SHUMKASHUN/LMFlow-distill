import jsonlines
import random
from torch.utils.data import Dataset

class TeacherDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = []
        with open(data_path, "r") as f:
            for item in jsonlines.Reader(f): 
                del item["logprobs"]["tokens"] # delete unused tokens
                del item["logprobs"]["token_logprobs"] # delete unused token_logprobs
                # del item["logprobs"]["top_log_probs"][0] # delete starting null value
                self.data.append(item)
        random.shuffle(self.data)  # shuffle the data after loading
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return{
            'text': self.data[index]['text'],
            'top_log_probs': self.data[index]["logprobs"]['top_log_probs']
        }
    
    # For each batch, there are three keys: input_token, output_token, top_logprobs
    # input_token: input token
    # output_token: output token
    # top_logprob: top 5 probability in log space
    def collate_fn(self, batch):
        return{
            'input_token': [x['text'] for x in batch],
            'output_token': [[[int(z) for z in list(y.keys())] for y in x['top_log_probs']] for x in batch],
            'top_logprob': [[list(y.values()) for y in x['top_log_probs']] for x in batch],
        }