"""
方案A（微调模型） + 方案B（FAISS 向量检索）级联评测脚本

思路（置信度阈值路由）：
  每条测试样本先走 FAISS 检索（方案B，快），拿到最近邻标签 + 余弦相似度 s。
    - s >= 阈值 T：检索有把握 → 直接用检索标签（快，不调模型）
    - s <  阈值 T：检索没把握 → 交给微调模型精判（方案A，准但慢）

为什么阈值挂在「相似度」而不是「标签」上：
  相似度才是检索的置信度——相似度高说明语义上找到了几乎一样的样本，
  标签大概率可信；相似度低说明是边缘样本，检索拿不准，才需要模型兜底。

运行（测试集 embedding 只算一次）：
  第一步（选阈值，不调模型，只看检索分布）：
      python cascade_evaluate.py --analyze
  第二步（按阈值跑完整级联，需先启动微调模型 API）：
      python cascade_evaluate.py --threshold 0.80
"""

import json
import re
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import faiss
from openai import OpenAI

import faiss_evaluate as fe

BASE_DIR = Path(__file__).parent
DETAILS_FILE = BASE_DIR / "cascade_details.jsonl"  # 检索详情（中间结果）

SYSTEM_PROMPT = "你是一名提示词安全审查员"

thread_local = threading.local()
_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 微调模型 API（复用 evaluate.py 的调用方式，保证与 v2 评测一致）
# ---------------------------------------------------------------------------
def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )
    return thread_local.client


def get_model():
    c = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    return c.models.list().data[0].id


def extract_judgment(text: str):
    """从模型回复中提取 safe / unsafe（取第一个命中，与 evaluate.py 一致）。"""
    t = text.strip().lower()
    if t == "safe":
        return "safe"
    if t == "unsafe":
        return "unsafe"
    m = re.findall(r"\b(safe|unsafe)\b", t)
    return m[0] if m else None


def model_classify(model: str, text: str) -> str:
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=16,
        temperature=0,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# 归一化 + 检索
# ---------------------------------------------------------------------------
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def build_normalized_index():
    """把已有 FAISS 索引里的向量重建 + 归一化，变成余弦相似度索引（本地操作，不调 API）。"""
    index = fe.faiss_load(fe.INDEX_FILE)
    n = index.ntotal
    raw = index.reconstruct_n(0, n).astype(np.float32)
    norm_idx = faiss.IndexFlatIP(raw.shape[1])
    norm_idx.add(l2_normalize(raw))
    return norm_idx


def run_retrieval():
    """对测试集做检索，返回 [{idx, true_label, pred_label, score, text}] 并落盘。"""
    index = build_normalized_index()
    with open(fe.LABELS_FILE, "r", encoding="utf-8") as f:
        train_labels = json.load(f)
    test_records = fe.read_jsonl(fe.TEST_FILE)

    results = [None] * len(test_records)
    done = 0
    total = len(test_records)

    def worker(i, rec):
        t0 = time.time()
        vec = fe.get_embedding(rec["text"])
        embed_time = time.time() - t0
        if vec is None:
            result = {"idx": i, "true_label": rec["label"], "pred_label": None,
                      "score": None, "text": rec["text"], "error": "向量化失败",
                      "embed_time": embed_time}
            return i, result
        q = l2_normalize(np.array([vec], dtype=np.float32))
        d, idx_arr = index.search(q, 1)
        score = float(d[0][0])
        pred = train_labels[idx_arr[0][0]]
        time.sleep(fe.REQ_DELAY)
        result = {"idx": i, "true_label": rec["label"], "pred_label": pred,
                  "score": score, "text": rec["text"], "embed_time": embed_time}
        return i, result

    with ThreadPoolExecutor(max_workers=fe.EMBED_WORKERS) as ex:
        futs = [ex.submit(worker, i, r) for i, r in enumerate(test_records)]
        for fut in as_completed(futs):
            i, r = fut.result()
            results[i] = r
            done += 1
            with _print_lock:
                s = r["score"] if r["score"] is not None else -1.0
                print(f"  [{i + 1:>4}] 真实={r['true_label']:6s}  "
                      f"检索={(r['pred_label'] or 'N/A'):6s} 相似度={s:.4f}  "
                      f"{done}/{total}")

    results = [r for r in results if r]
    with open(DETAILS_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已保存检索详情到 {DETAILS_FILE}（{len(results)} 条）")
    return results


# ---------------------------------------------------------------------------
# 分析：不同阈值下的检索子集表现（用于选 T）
# ---------------------------------------------------------------------------
def analyze(results):
    valid = [r for r in results if r["score"] is not None]
    print("\n" + "=" * 78)
    print("不同阈值 T 下的路由情况（假设被路由给模型的样本准确率为 86%）")
    print("=" * 78)
    print(f"{'阈值T':>6} | {'检索占比':>8} | {'检索子集准确率':>12} | {'估算级联准确率':>12}")
    print("-" * 78)

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    for T in thresholds:
        confident = [r for r in valid if r["score"] >= T]
        frac = len(confident) / len(valid)
        acc_c = sum(1 for r in confident if r["pred_label"] == r["true_label"]) / len(confident) if confident else 0
        est = frac * acc_c + (1 - frac) * 0.86  # 模型部分按历史 86% 估
        print(f"{T:>6.2f} | {frac * 100:>7.1f}% | {acc_c * 100:>12.2f}% | {est * 100:>12.2f}%")
    print("=" * 78)
    print("提示：选一个「检索占比大 + 检索子集准确率 >= 86%」的 T，级联才有提升。")
    print("然后运行：python cascade_evaluate.py --threshold <你选的T>")


# ---------------------------------------------------------------------------
# 完整级联：阈值路由 + 微调模型兜底
# ---------------------------------------------------------------------------
def run_cascade(T: float):
    with open(DETAILS_FILE, "r", encoding="utf-8") as f:
        details = [json.loads(l) for l in f if l.strip()]
    details = [r for r in details if r["score"] is not None]  # 去掉向量化失败的

    by_retrieval = [r for r in details if r["score"] >= T]
    by_model = [r for r in details if r["score"] < T]
    total = len(details)

    print(f"\n阈值 T = {T}")
    print(f"路由给检索（s >= T）: {len(by_retrieval)} 条 ({len(by_retrieval) / total * 100:.1f}%)")
    print(f"路由给模型（s <  T）: {len(by_model)} 条 ({len(by_model) / total * 100:.1f}%)")

    # 检索那部分直接判
    retr_correct = sum(1 for r in by_retrieval if r["pred_label"] == r["true_label"])

    # 模型那部分并发调用
    model = get_model()
    model_result = {}
    model_total_time = 0.0
    done = 0

    def worker(r):
        t0 = time.time()
        pred = extract_judgment(model_classify(model, r["text"]))
        dur = time.time() - t0
        ok = (pred == r["true_label"])
        return r["idx"], ok, pred, dur

    print("\n模型精判中（只判相似度低的部分）...")
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(worker, r) for r in by_model]
        for fut in as_completed(futs):
            idx, ok, pred, dur = fut.result()
            model_result[idx] = ok
            model_total_time += dur
            done += 1
            with _print_lock:
                print(f"  [{idx + 1:>4}] 模型判={pred or 'N/A':6s} "
                      f"{'✓' if ok else '✗'}  {done}/{len(by_model)}")

    model_correct = sum(1 for r in by_model if model_result[r["idx"]])
    correct = retr_correct + model_correct

    # ---- 耗时统计 ----
    total_embed_time = sum(r.get("embed_time", 0.0) for r in details)
    cascade_total = total_embed_time + model_total_time  # 级联合计

    print("\n" + "=" * 70)
    print("级联评测汇总")
    print("=" * 70)
    print(f"测试集样本数:      {total}")
    print(f"检索直接判样本数:  {len(by_retrieval)}")
    print(f"模型兜底判样本数:  {len(by_model)}")
    print(f"正确数:            {correct}")
    print(f"整体准确率:        {correct / total * 100:.2f}%")
    print(f"  - 检索直接判:    {retr_correct}/{len(by_retrieval)} = "
          f"{retr_correct / len(by_retrieval) * 100:.2f}%")
    print(f"  - 模型兜底判:    {model_correct}/{len(by_model)} = "
          f"{model_correct / len(by_model) * 100:.2f}%")
    print()
    print("--- 耗时统计 ---")
    print(f"测试集 embedding:  {total_embed_time:.2f}s")
    print(f"模型精判:          {model_total_time:.2f}s")
    print(f"级联合计:          {cascade_total:.2f}s")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="方案A+方案B 级联评测")
    parser.add_argument("--analyze", action="store_true",
                        help="只做检索并分析阈值分布（不调模型）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="完整级联使用的余弦相似度阈值（如 0.80）")
    args = parser.parse_args()

    if not args.analyze and args.threshold is None:
        parser.error("请指定 --analyze 或 --threshold T")

    if args.analyze:
        results = run_retrieval()
        analyze(results)
    else:
        run_cascade(args.threshold)


if __name__ == "__main__":
    main()