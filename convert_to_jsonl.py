import json
from datasets import load_dataset

base_dir = r"d:\AITest\aitest\Prompt安全检测数据集"

SYSTEM_PROMPT = "你是一名提示词安全审查员"

# 训练集需要精简到 30000 条（15000 safe + 15000 unsafe）
TRAIN_TARGET_PER_CLASS = 15000
# 去除过短的无意义文本
MIN_TEXT_LENGTH = 5

files = {
    "train": f"{base_dir}/train.parquet",
    "val":   f"{base_dir}/val.parquet",
    "test":  f"{base_dir}/test.parquet",
}

for name, path in files.items():
    print(f"正在处理 {name} ...")
    dataset = load_dataset("parquet", data_files={name: path})[name]
    # 只保留中文数据
    dataset = dataset.filter(lambda x: x["lang"] == "zh")

    rows = list(dataset)

    if name == "train":
        # 过滤过短文本
        before_filter = len(rows)
        rows = [r for r in rows if len(r["text"]) > MIN_TEXT_LENGTH]
        after_filter = len(rows)
        print(f"  过滤短文本: {before_filter} -> {after_filter} (移除 {before_filter - after_filter} 条)")

        # 分层采样
        safe_entries   = [r for r in rows if r["label"] == "safe"]
        unsafe_entries = [r for r in rows if r["label"] == "unsafe"]
        print(f"  safe={len(safe_entries)}, unsafe={len(unsafe_entries)}")

        import random
        random.seed(42)
        sampled_safe   = random.sample(safe_entries,   min(TRAIN_TARGET_PER_CLASS, len(safe_entries)))
        sampled_unsafe = random.sample(unsafe_entries, min(TRAIN_TARGET_PER_CLASS, len(unsafe_entries)))
        rows = sampled_safe + sampled_unsafe
        random.shuffle(rows)
        print(f"  采样后: {len(rows)} 条 (safe={len(sampled_safe)}, unsafe={len(sampled_unsafe)})")

    output_path = f"{base_dir}/sft_{name}_zh.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            messages = [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": row["text"]},
                {"role": "assistant", "content": row["label"]},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"  -> 完成: {output_path} ({len(rows)} 条)")
