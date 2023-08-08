"""
This file is to combine multi-thread generated soft probability example files in to one final file
The count for robin training data of block size 512 is around 31w, and for block size 2048 is 8w
"""


import os
import pandas
import sys
import json
import jsonlines
count = 0
for root, ds,fs in os.walk('./DistilledData/'):
    for f in fs:
        # print(os.path.join(root,f))
        # if f.endswith(".jsonl"):
        list_a = []
        with open(os.path.join(root,f),"r+") as file:
            for item in jsonlines.Reader(file):
                list_a.append(item)
        output_writer = jsonlines.open("./distilled_data.jsonl", "a")
        for index, output in enumerate(list_a):
            output_writer.write(output)
        count += len(list_a)
        print(f"{f} processed")
print(count)