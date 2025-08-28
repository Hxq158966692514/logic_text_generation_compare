import json
import pandas as pd
import torch
from datasets import Dataset, load_from_disk


PROMPT = "你是一个数学专家，你需要根据用户的问题，给出带有思考的问题答案。"

# 进行数据处理

data = load_from_disk('less is more for reasoning/less is more for reasoning')['train']

df = data.train_test_split(test_size=0.5,shuffle=True)

train_dataset = df['train']

test_dataset = df['test']

# print(train_dataset['question'][0])
# print('*'*100)
# print(train_dataset['solution'][0])
# print('*'*100)
# print(train_dataset['answer'][0])
#
# print(test_dataset)

# 保存训练集
with open('less is more for reasoning_ORI/train.jsonl', 'w', encoding='utf-8') as f:
    for index in range(len(train_dataset['question'])):
        json.dump({"instruction":PROMPT,'question':train_dataset['question'][index],'solution':train_dataset['solution'][index],'answer':train_dataset['answer'][index]}, f, ensure_ascii=False)
        f.write('\n')

# 保存验证集
with open('less is more for reasoning_ORI/val.jsonl', 'w', encoding='utf-8') as f:
    for index in range(len(test_dataset['question'])):
        json.dump({"instruction":PROMPT,'question':test_dataset['question'][index],'solution':test_dataset['solution'][index],'answer':test_dataset['answer'][index]}, f, ensure_ascii=False)
        f.write('\n')