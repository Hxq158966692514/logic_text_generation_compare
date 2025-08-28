import json
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
from peft import LoraConfig,TaskType,get_peft_model
from transformers import BitsAndBytesConfig

import numpy as np

import evaluate

import math

PROMPT = "你是一个数学专家，你需要根据用户的问题，给出带有思考的问题答案。"
MAX_LENGTH = 2048

bleu = evaluate.load("sacrebleu")

rouge = evaluate.load("rouge")

def compute_metrics(pred):

    labels_ids = pred.label_ids

    pred_ids = pred.predictions[0]

    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)

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

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input = data["question"]
            think = data["solution"]
            answer = data["answer"]
            output = f"<solution>{think}</solution> \n {answer}"  # 这种结合方式非常棒
            message = {
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    将数据集进行预处理
    """
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  # 输入模块不进行训练（不参与）
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("Qwen1.5-4B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen1.5-4B", device_map="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True))
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 配置lora
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

# 加载、处理数据集和测试集
train_dataset_path = "less is more for reasoning_ORI/train.jsonl"
test_dataset_path = "less is more for reasoning_ORI/val.jsonl"

train_jsonl_new_path = "less is more for reasoning_ORI/train_format.jsonl"
test_jsonl_new_path = "less is more for reasoning_ORI/val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)  # 数据类型的转换
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 得到验证集
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

args = TrainingArguments(
    output_dir="./output/Qwen1.5-4B",
    per_device_train_batch_size= 3,
    per_device_eval_batch_size= 3,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=15,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=15,
    learning_rate=2e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    run_name="qwen1.5-4B",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

trainer.train()

# 测试

metrics = trainer.evaluate()

ppl = math.exp(min(20, metrics["eval_loss"]))

print(metrics)

print({'Perplexity':ppl})

# 用测试集的前3条，主观看模型
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]

test_text_list = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)

    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """

    test_text_list.append(response_text)
    print(response_text)