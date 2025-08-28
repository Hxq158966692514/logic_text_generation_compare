import os
import math
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer

# ---------------- 参数 ----------------
MODEL_NAME = "Qwen1.5-4B"
OUTPUT_DIR = "qwen15-sft-metrics"
TRAIN_FILE = "less is more for reasoning_ORI/train.jsonl"
EVAL_FILE = "less is more for reasoning_ORI/eval.jsonl"

MAX_LEN = 1024

# ---------------- 加载模型和分词器 ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# ---------------- 数据 ----------------
raw_datasets = load_dataset("json", data_files={"train": TRAIN_FILE, "eval": EVAL_FILE})

def format_example(ex):
    # 简化：instruction + output 格式
    return {"text": f"用户: {ex['instruction']}\n助手: {ex['output']}"}

train_dataset = raw_datasets["train"].map(format_example, remove_columns=raw_datasets["train"].column_names)
eval_dataset = raw_datasets["eval"].map(format_example, remove_columns=raw_datasets["eval"].column_names)

# ---------------- 评估方法 ----------------
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    """
    eval_preds: EvalPrediction(predictions, label_ids)
    - predictions: 模型生成的 token 序列 (logits argmax or generate)
    - label_ids: 参考答案的 token 序列
    """
    preds, labels = eval_preds
    # 转为文本
    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU
    bleu_result = bleu.compute(predictions=pred_texts, references=[[l] for l in label_texts])
    # ROUGE
    rouge_result = rouge.compute(predictions=pred_texts, references=label_texts)

    # PPL = exp(loss)，SFTTrainer 会在 metrics 里给 'eval_loss'
    # 这里预留：如果外部传了 loss，就算；否则 NaN
    ppl = float("nan")
    if hasattr(eval_preds, "metrics") and "eval_loss" in eval_preds.metrics:
        ppl = math.exp(min(20, eval_preds.metrics["eval_loss"]))

    return {
        "bleu": bleu_result["score"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "rougeLsum": rouge_result["rougeLsum"],
        "ppl": ppl
    }

# ---------------- 训练 ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    report_to=["none"]
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_LEN,
    args=training_args,
    compute_metrics=compute_metrics,   # 👈 挂载 metrics
    packing=False
)

#trainer.train()
metrics = trainer.evaluate()
print(metrics)
