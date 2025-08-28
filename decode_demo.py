from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen1.5-4B", use_fast=False)

# 模拟两条 token 序列
batch_ids = [
    [20005, 195, 318, 123, tokenizer.eos_token_id],   # 一句话
    [195, 276, 318, 623, tokenizer.eos_token_id],     # 另一句话
]

# 转成文本
decoded_texts = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)

print(decoded_texts)
