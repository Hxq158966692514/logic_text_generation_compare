#-*- codeing = utf-8 -*-
#@Time : 2025-08-22 23:46
#@Author : 韩笑奇
#@File : test_demo.py
#@Software: PyCharm

from transformers import pipeline

from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer

from datasets import load_from_disk,Dataset,DatasetDict

import pandas as pd


# firefly 文本生成
# model = AutoPeftModelForCausalLM.from_pretrained('./experiments/final_result',device_map='cpu')
#
# tokenizer = AutoTokenizer.from_pretrained('Qwen1.5-4B\Qwen2.5-3B.txt-4B')
#
# tokenizer.padding_side = 'right'
#
# merged_model = model.merge_and_unload()
#
# pip = pipeline(task='text-generation',model=merged_model,tokenizer=tokenizer)
#
# prompt = '<|im_start|>system\n你是一个很厉害的ai助手,具有一定的逻辑推理能力！！<|im_end|>\n<|im_start|>user\n阅读文章，回答问题：\n枪鱼价格浮动貌似比较大，相对贵一些。金枪鱼零下20度的是160元每公斤，零下60度的是380元每公斤。所有的三文鱼都是冰鲜的(速冻的)，价格要看三文鱼的产地。超市的价格是 ：---挪威 (零售价约每公斤130元)---加拿大(零售价约每公斤145元)----丹麦 (零售价约每公斤150元)\n问题：金枪鱼和三文鱼哪个贵<|im_end|>\n<|im_start|>assistant\n'
#
# print(pip(prompt,max_new_tokens=50)[0]['generated_text'])


#less is more for reasoning   文本生成

model = AutoPeftModelForCausalLM.from_pretrained('./experiments/final_result_reasoning_3',device_map='cpu')
#
tokenizer = AutoTokenizer.from_pretrained('Qwen1.5-4B')

tokenizer.padding_side = 'right'

merged_model = model.merge_and_unload()

pip = pipeline(task='text-generation',model=merged_model,tokenizer=tokenizer)

# data = load_from_disk('less is more for reasoning/less is more for reasoning')['train']
#
# df = data.train_test_split(test_size=0.001,shuffle=True)
#
# test_data = df['test']

# print(test_data)

df = pd.read_csv('test_data.csv')
# print(df)
test_mid = {
    'question': [str(df['question'].iloc[0])],
    'solution': [str(df['solution'].iloc[0])],
    'answer': [str(df['answer'].iloc[0])],
}

# print(type(test_mid))

test_data = Dataset.from_dict(test_mid)

# print(test_data)



def format_prompt_reasoning(example):

    chat = [

        {'role':'system','content':'我是一个智能ai助手, 能计算较为复杂的算术题！！'},

        {'role':'user','content':example['question']+'\n'+example['solution']},

        {'role':'assistant','content':example['answer']}

    ]

    prompt = tokenizer.apply_chat_template(chat,tokenize=False)

    return {'text':prompt}

dataset = test_data.map(format_prompt_reasoning,remove_columns=test_data.column_names)

print(dataset[-1]['text'])

prompt_list = dataset[-1]['text'].split('\n')[:-2]

prompt = ''

for i in prompt_list:

    prompt+=i

    prompt+='\n'


print(pip(prompt,max_new_tokens=50)[0]['generated_text'])




