#-*- codeing = utf-8 -*-
#@Time : 2025-08-24 8:45
#@Author : 韩笑奇
#@File : train_SFT.py
#@Software: PyCharm


from datasets import load_from_disk

from transformers import AutoTokenizer

from peft import LoraConfig, get_peft_model,TaskType

from transformers import TrainingArguments

from transformers import AutoModelForCausalLM, Trainer

from transformers import BitsAndBytesConfig

import evaluate

import numpy as np

import math

import torch


# 加载 firefly数据集
# data = load_from_disk('data')['train']
#
# df = data.train_test_split(test_size=0.999,shuffle=True)
#
# train_data = df['train']
#
# print(train_data[100])

# 加载less is more for reasoning 数据集

data = load_from_disk('less is more for reasoning/less is more for reasoning')['train']

df = data.train_test_split(test_size=0.5,shuffle=True)

print(df)

train_data = df['train']

test_data = df['test']

df['test'].to_csv('test_data.csv')

# print(train_data[100])


# 转换为千问模型能够处理的数据格式（founation_model）

tokenizer = AutoTokenizer.from_pretrained('Qwen1.5-4B',use_fast=False,trust_remote_code=True)

tokenizer.padding_side = 'right'

bleu = evaluate.load("sacrebleu")

rouge = evaluate.load("rouge")


def compute_metrics(pred):

    labels_ids = pred.label_ids

    pred_ids = pred.predictions[0]

    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)

    labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)

    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    label_texts = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # BLEU
    bleu_result = bleu.compute(predictions=pred_texts, references=[[l] for l in label_texts])
    # ROUGE
    rouge_result = rouge.compute(predictions=pred_texts, references=label_texts)


    return {
        "bleu": bleu_result["score"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "rougeLsum": rouge_result["rougeLsum"],
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels



# 适用于firefly数据集的token
# def format_prompt(example):
#
#     chat = [
#
#         {'role':'system','content':'你是一个很厉害的ai助手,具有一定的逻辑推理能力！！'},
#
#         {'role':'user','content':example['input']},
#
#         {'role':'assistant','content':example['target']}
#
#     ]
#
#     prompt = tokenizer.apply_chat_template(chat,tokenize=False)
#
#     return {'text':prompt}
#
# dataset = train_data.map(format_prompt,remove_columns=train_data.column_names)
#
# print(dataset[0])

# less is more for reasoning 数据格式整合

print(type(train_data[0]['solution']))

def format_prompt_reasoning(example):

    chat = [

        {'role':'system','content':'这是一个智能数学天才！！'},

        {'role':'user','content':example['question']},

        {'role':'assistant','content':example['solution']+'\n'+example['answer']}

    ]

    prompt = tokenizer.apply_chat_template(chat,tokenize=False)

    return {'text':prompt}

train_dataset = train_data.map(format_prompt_reasoning,remove_columns=train_data.column_names)

test_dataset = test_data.map(format_prompt_reasoning,remove_columns=test_data.column_names)

print(train_dataset[0])

model = AutoModelForCausalLM.from_pretrained('Qwen1.5-4B',quantization_config=BitsAndBytesConfig(load_in_8bit=True),device_map='auto')
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
# 书写lora配置，目的是为了参数精简。

peft_config = LoraConfig(

    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例

)

model = get_peft_model(model,peft_config)


# 进行训练参数的加载

output_dir = './experiments'

# 默认训练器
# training_arguments = TrainingArguments(
#
#     output_dir = output_dir,
#
#     per_device_train_batch_size = 2,
#
#     gradient_accumulation_steps = 4,
#
#     optim = 'adamw_torch',
#
#     learning_rate = 2e-4,
#
#     lr_scheduler_type = 'cosine',
#
#     num_train_epochs = 2,
#
#     logging_steps = 10,
#
#     fp16 = True,
#
#     gradient_checkpointing = True,
#
#     save_steps = 15,      #15 个step 就保存一个 checkpoint
#
#     max_steps = 20,
#
#     remove_unused_columns= False
#
# )

from trl import SFTTrainer, SFTConfig

#   自带训练器
training_arguments = SFTConfig(

    output_dir = output_dir,

    per_device_train_batch_size = 3,

    per_device_eval_batch_size = 3,

    gradient_accumulation_steps = 4,

    eval_accumulation_steps = 1,

    optim = 'adamw_torch',

    learning_rate = 2e-4,

    lr_scheduler_type = 'cosine',

    num_train_epochs = 3,

    logging_steps = 10,

    fp16 = True,

    gradient_checkpointing = True,

    save_steps = 15,      #15 个step 就保存一个 checkpoint

    max_steps = 20,

    remove_unused_columns= False,

    packing = True

)

trainer = SFTTrainer(

    model = model,

    args = training_arguments,

    train_dataset = train_dataset,

    eval_dataset = test_dataset,

    peft_config = peft_config,

    compute_metrics = compute_metrics,

    preprocess_logits_for_metrics = preprocess_logits_for_metrics


)

trainer.train()

metrics = trainer.evaluate()

ppl = math.exp(min(20, metrics["eval_loss"]))

print(metrics)

print({'Perplexity':ppl})

#保存firefly数据集模型结果
#trainer.model.save_pretrained('./experiments/final_result')

#保存less is more for reasoning数据集结果

trainer.model.save_pretrained('./experiments/final_result_reasoning_qwen1.5_4b')

# 查看参数量化效果

print(trainer.model.print_trainable_parameters())



