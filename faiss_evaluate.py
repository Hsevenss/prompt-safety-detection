"""
FAISS 检索评测脚本（训练/测试分离版）

流程：
  1. 读取 sft_train_zh.jsonl（30k 条）→ 向量化 → 建 FAISS 索引并存盘
  2. 读取 sft_test_zh.jsonl（3195 条）→ 向量化 → 检索 k=1
  3. 对比预测标签与真实标签，统计准确率 + 耗时

特点：
  - 多线程并发 + 速率控制，避免被 API 断连
  - 增量保存索引，支持断点续传
  - 失败自动重试（指数退避）
"""

import json
import time
import ssl
import argparse
import random
import requests
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import faiss


# ---------------------------------------------------------------------------
# SSL & Session
# ---------------------------------------------------------------------------
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers('DEFAULT:@SECLEVEL=1')
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


_session = None
_session_lock = Lock()


def get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                _session.mount('https://', SSLAdapter())
    return _session


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
API_KEY = "902539cd5d604193b6aac2795385abd6.sNO6QufedXIRpfZu"
EMBEDDING_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
EMBEDDING_MODEL = "embedding-3"

BASE_DIR = Path(__file__).parent
TRAIN_FILE = BASE_DIR / "sft_train_zh.jsonl"
TEST_FILE  = BASE_DIR / "sft_test_zh.jsonl"

# FAISS 索引及相关文件（持久化，支持断点续传）
INDEX_FILE     = BASE_DIR / "faiss_index.bin"
LABELS_FILE    = BASE_DIR / "train_labels.json"
# 检查点文件：记录已向量化的训练数据行号（断点续传用）
CHECKPOINT_FILE = BASE_DIR / "embed_checkpoint.txt"

# 并发 & 速率控制
EMBED_WORKERS   = 20     # 向量化并发线程数
REQ_DELAY       = 0.1     # 每次请求后休眠秒数（防限流）
MAX_RETRIES     = 3       # 单次请求最大重试次数
CHECKPOINT_STEP = 500     # 每 N 条保存一次检查点


# ---------------------------------------------------------------------------
# FAISS 读写辅助（中文路径兼容：Windows 下 faiss 的 C++ fopen 打不开含
# 中文/非 ASCII 字符的路径，改用内存序列化 + Python 自己的文件 IO 绕开）
# ---------------------------------------------------------------------------
def faiss_load(path):
    with open(path, "rb") as f:
        return faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))


def faiss_save(index, path):
    with open(path, "wb") as f:
        f.write(faiss.serialize_index(index).tobytes())


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def get_embedding(text: str, retry: int = MAX_RETRIES) -> list[float] | None:
    """调用智谱 embedding API，失败时指数退避重试。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }
    for attempt in range(1, retry + 1):
        try:
            resp = get_session().post(EMBEDDING_URL, headers=headers,
                                      json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["data"][0]["embedding"]
            else:
                print(f"  [WARN] API 返回 {resp.status_code}, 第 {attempt} 次重试...")
        except Exception as e:
            print(f"  [WARN] 请求异常: {e}, 第 {attempt} 次重试...")
        # 指数退避 + 随机抖动
        time.sleep(2 ** attempt + random.uniform(0, 1))
    return None


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def read_jsonl(file_path: Path) -> list[dict]:
    """读取 JSONL 文件，返回 [{'text': ..., 'label': ...}, ...] 列表。"""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = data.get("messages", [])
            user_content = None
            label = None
            for msg in messages:
                if msg["role"] == "user":
                    user_content = msg["content"]
                elif msg["role"] == "assistant":
                    label = msg["content"]
            if user_content and label:
                records.append({"text": user_content, "label": label})
    return records


# ---------------------------------------------------------------------------
# 读取检查点
# ---------------------------------------------------------------------------
def read_checkpoint() -> int:
    """返回已完成的条数（行号偏移），没有则返回 0。"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def write_checkpoint(done: int):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(done))


# ---------------------------------------------------------------------------
# 多线程向量化 + 建索引
# ---------------------------------------------------------------------------
def embed_records(records: list[dict], start_offset: int = 0) -> tuple:
    """
    多线程向量化所有 records，返回 (vectors, labels, total_time)。
    start_offset: 已完成的条数（用于进度显示）。
    """
    total = len(records)
    if total == 0:
        return [], [], 0.0

    vectors = [None] * total  # 预分配，保证结果顺序
    labels  = [None] * total
    fail_count = 0
    embed_total_time = 0.0
    done_count = start_offset
    lock = Lock()

    def worker(idx: int, text: str, label: str):
        nonlocal fail_count, embed_total_time, done_count
        t0 = time.time()
        vec = get_embedding(text)
        elapsed = time.time() - t0

        with lock:
            embed_total_time += elapsed
            if vec is None:
                fail_count += 1
                vectors[idx] = None
            else:
                vectors[idx] = vec
                labels[idx] = label
            done_count += 1

            curr = done_count
            # 每 CHECKPOINT_STEP 条或完成时打印进度
            if curr % CHECKPOINT_STEP == 0 or curr == start_offset + total:
                success = curr - start_offset - fail_count
                print(f"  进度: {curr}/{start_offset + total} 条 "
                      f"(成功={success}, 失败={fail_count}, "
                      f"平均耗时={embed_total_time / (curr - start_offset):.3f}s)")

        # 请求间延迟（防限流）
        time.sleep(REQ_DELAY)

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        futures = [
            executor.submit(worker, i, r["text"], r["label"])
            for i, r in enumerate(records)
        ]
        for f in as_completed(futures):
            f.result()  # 触发异常（如果有）

    # 过滤失败的
    valid_vectors = []
    valid_labels = []
    for v, l in zip(vectors, labels):
        if v is not None:
            valid_vectors.append(v)
            valid_labels.append(l)

    return valid_vectors, valid_labels, embed_total_time


# ---------------------------------------------------------------------------
# 建 FAISS 索引（支持增量合并）
# ---------------------------------------------------------------------------
def build_index(vectors: list, labels: list, append: bool = False) -> tuple:
    """
    建 FAISS 索引。
    若 append=True 且已有索引文件，则先加载旧索引再追加新向量。
    返回 (index, labels)。
    """
    dim = len(vectors[0])
    matrix = np.array(vectors).astype(np.float32)

    if append and INDEX_FILE.exists():
        print(f"  加载已有索引 ({INDEX_FILE})，追加新向量...")
        index = faiss_load(INDEX_FILE)
        index.add(matrix)
        # 加载已有标签
        old_labels_path = LABELS_FILE
        if old_labels_path.exists():
            with open(old_labels_path, "r", encoding="utf-8") as f:
                old_labels = json.load(f)
            labels = old_labels + labels
        print(f"  合并后索引大小: {index.ntotal}")
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

    faiss_save(index, INDEX_FILE)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f)

    print(f"  索引已保存: {INDEX_FILE} ({index.ntotal} 个向量)")
    return index, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FAISS 检索评测（训练/测试分离）")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重建训练集索引（忽略检查点）")
    args = parser.parse_args()

    # ==================================================================
    # Phase 1: 读训练数据 → 向量化 → 建索引
    # ==================================================================
    print("=" * 70)
    print("Phase 1: 读取训练数据 & 建 FAISS 索引")
    print("=" * 70)

    train_records = read_jsonl(TRAIN_FILE)
    total_train = len(train_records)
    print(f"训练集: {total_train} 条")

    # 检查断点
    checkpoint = 0 if args.rebuild else read_checkpoint()
    if checkpoint > 0 and not args.rebuild:
        print(f"检测到检查点: 已完成 {checkpoint}/{total_train} 条，从中断处继续")
    elif args.rebuild:
        print("强制重建模式，忽略已有检查点")

    remaining = train_records[checkpoint:]

    if remaining:
        print(f"需要向量化: {len(remaining)} 条")
        print()

        vectors, labels, _ = embed_records(remaining, start_offset=checkpoint)

        if len(vectors) == 0:
            print("[ERROR] 向量化全部失败，退出")
            return

        # 建索引（追加模式：第一次用 append=False，后续按需传 True）
        is_append = checkpoint > 0 and INDEX_FILE.exists()
        build_index(vectors, labels, append=is_append)

        # 更新检查点
        write_checkpoint(total_train)
        print(f"  检查点已更新: {CHECKPOINT_FILE}")
    else:
        print("训练数据已全部向量化，直接加载已有索引")
        if not INDEX_FILE.exists():
            print("[ERROR] 索引文件不存在，请去掉 --rebuild 或删除检查点后重试")
            return

    print(f"\n最终索引大小: {faiss_load(INDEX_FILE).ntotal}")
    print()

    # ==================================================================
    # Phase 2: 读测试数据 → 向量化 → FAISS 检索
    # ==================================================================
    print("=" * 70)
    print("Phase 2: 测试集向量化 & FAISS 检索 (k=1)")
    print("=" * 70)

    test_records = read_jsonl(TEST_FILE)
    total_test = len(test_records)
    print(f"测试集: {total_test} 条\n")

    # 加载最终索引（在线程间共享，只读安全）
    index = faiss_load(INDEX_FILE)
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        train_labels = json.load(f)

    phase2_results = []
    phase2_lock = Lock()

    def query_worker(idx: int, text: str, true_label: str) -> dict:
        """单条查询：向量化 + FAISS 检索，返回结果字典。"""
        result = {
            "idx": idx,
            "true_label": true_label,
            "pred_label": None,
            "score": None,
            "embed_time": 0,
            "retrieve_time": 0,
            "total_time": 0,
            "is_correct": False,
            "error": None,
        }

        t0 = time.time()
        query_vec = get_embedding(text)
        embed_time = time.time() - t0
        result["embed_time"] = embed_time

        if query_vec is None:
            result["error"] = "向量化失败"
            return result

        query_np = np.array([query_vec]).astype(np.float32)

        t1 = time.time()
        distances, indices = index.search(query_np, 1)
        retrieve_time = time.time() - t1
        result["retrieve_time"] = retrieve_time
        result["total_time"] = embed_time + retrieve_time

        nearest_idx = indices[0][0]
        score = distances[0][0]
        pred_label = train_labels[nearest_idx]

        result["score"] = score
        result["pred_label"] = pred_label
        result["is_correct"] = (pred_label == true_label)

        time.sleep(REQ_DELAY)  # 防限流
        return result

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        futures = {
            executor.submit(query_worker, i, rec["text"], rec["label"]): i
            for i, rec in enumerate(test_records)
        }
        for future in as_completed(futures):
            r = future.result()
            phase2_results.append(r)

            # 实时打印
            status = "✓" if r["is_correct"] else "✗" if not r["error"] else "?"
            with phase2_lock:
                if r["error"]:
                    print(f"  [{r['idx'] + 1:>4}] {status}  真实={r['true_label']:6s}  "
                          f"错误: {r['error']}")
                else:
                    print(f"  [{r['idx'] + 1:>4}] {status}  "
                          f"真实={r['true_label']:6s} 预测={r['pred_label']:6s}  "
                          f"相似度={r['score']:.4f}  "
                          f"embed={r['embed_time']:.3f}s "
                          f"检索={r['retrieve_time']:.4f}s "
                          f"合计={r['total_time']:.3f}s")

    # 统计
    valid_results = [r for r in phase2_results if not r["error"]]
    total_done = len(valid_results)
    correct = sum(1 for r in valid_results if r["is_correct"])
    embed_query_times = [r["embed_time"] for r in valid_results]
    retrieve_times = [r["retrieve_time"] for r in valid_results]
    total_failed = sum(1 for r in phase2_results if r["error"])

    # ==================================================================
    # Summary
    # ==================================================================
    print()
    print("=" * 70)
    print("汇总结果")
    print("=" * 70)

    accuracy = correct / total_done * 100 if total_done > 0 else 0
    avg_embed_time = np.mean(embed_query_times) if embed_query_times else 0
    avg_retrieve_time = np.mean(retrieve_times) if retrieve_times else 0
    avg_total_time = avg_embed_time + avg_retrieve_time

    print(f"训练集索引向量数: {index.ntotal}")
    print(f"测试集样本数:     {total_done}")
    print(f"正确数:           {correct}")
    print(f"准确率:           {accuracy:.2f}%")
    print(f"向量化失败:       {total_failed}")
    print(f"--- 耗时统计（单条）---")
    print(f"平均向量化:       {avg_embed_time:.3f}s")
    print(f"平均检索:         {avg_retrieve_time * 1000:.2f}ms")
    print(f"平均合计:         {avg_total_time:.3f}s")
    print(f"测试集总耗时:     {sum(embed_query_times) + sum(retrieve_times):.2f}s")


if __name__ == "__main__":
    main()
