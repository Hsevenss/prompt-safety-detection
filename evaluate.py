"""
评测脚本：读取 sft_test_zh.jsonl，多线程并发调用 API，实时展示结果并汇总准确率。

用法：
    python evaluate.py                          # 评测全部 3195 条
    python evaluate.py --max-samples 10         # 只测前 10 条
    python evaluate.py --label unsafe           # 只测 unsafe 类
    python evaluate.py --max-samples 10 --label safe
"""

import json
import re
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

DATA_FILE = Path(__file__).parent / "sft_test_zh.jsonl"

# 线程局部存储，避免多线程共用同一个 client
thread_local = threading.local()


def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(
            api_key='EMPTY',
            base_url='http://localhost:8000/v1',
        )
    return thread_local.client


def get_model():
    """获取模型 ID（只调用一次，使用主线程 client）。"""
    c = OpenAI(api_key='EMPTY', base_url='http://localhost:8000/v1')
    return c.models.list().data[0].id


def extract_judgment(text: str) -> str | None:
    """从模型回复中提取 safe / unsafe 判断。"""
    text_clean = text.strip().lower()
    if text_clean == 'safe':
        return 'safe'
    if text_clean == 'unsafe':
        return 'unsafe'
    matches = re.findall(r'\b(safe|unsafe)\b', text_clean)
    if matches:
        return matches[0]
    return None


def evaluate_one(line_idx: int, line: str) -> dict:
    """处理单条样本，返回结果字典。"""
    result = {
        'line': line_idx + 1,
        'expected': None,
        'judgment': None,
        'is_correct': False,
        'error': None,
        'response_text': '',
        'user_preview': '',
        'duration': 0,
    }

    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        result['error'] = f'JSON 解析失败: {e}'
        return result

    messages_raw = data.get('messages', [])
    system_content = None
    user_content = None
    expected = None
    for msg in messages_raw:
        if msg['role'] == 'system':
            system_content = msg['content']
        elif msg['role'] == 'user':
            user_content = msg['content']
        elif msg['role'] == 'assistant':
            expected = msg['content']

    if not user_content or not expected:
        result['error'] = '缺少 user 或 assistant 字段'
        return result

    result['expected'] = expected
    result['user_preview'] = user_content

    # 构造 messages
    messages = []
    if system_content:
        messages.append({'role': 'system', 'content': system_content})
    messages.append({'role': 'user', 'content': user_content})

    # 调用 API
    client = get_client()
    t0 = time.time()
    try:
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
        response_text = resp.choices[0].message.content or ''
    except Exception as e:
        result['error'] = f'API 调用失败: {e}'
        result['duration'] = time.time() - t0
        return result

    result['duration'] = time.time() - t0
    result['response_text'] = response_text

    judgment = extract_judgment(response_text)
    result['judgment'] = judgment

    if judgment is None:
        result['is_correct'] = False
    else:
        result['is_correct'] = (judgment == expected)

    return result


# 全局变量（主线程持有），用于实时打印
_print_lock = threading.Lock()


def print_result(result: dict):
    """线程安全地打印单条结果。"""
    status_icon = "✓" if result['is_correct'] else "✗" if result['judgment'] else "?"
    expected = result['expected'] or "-"
    judgment = result['judgment'] or "N/A"
    duration = result['duration']
    error = result.get('error')

    with _print_lock:
        if error:
            print(f"[{result['line']:>4}] {status_icon}  | 期望={expected}  | 错误: {error}")
        else:
            print(
                f"[{result['line']:>4}] {status_icon}  "
                f"期望={expected}  模型={judgment}  "
                f"耗时={duration:.2f}s"
            )
            # 只显示部分 response 方便对照
            resp_short = result['response_text'].replace('\n', ' ').strip()[:120]
            # print(f"      回应: {resp_short}")


def evaluate(max_samples: int | None = None, filter_label: str | None = None):
    # 读取所有样本
    all_lines = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if filter_label:
                try:
                    data = json.loads(line)
                    expected = None
                    for msg in data.get('messages', []):
                        if msg['role'] == 'assistant':
                            expected = msg['content']
                    if expected != filter_label:
                        continue
                except json.JSONDecodeError:
                    continue
            all_lines.append((line_idx, line))
            if max_samples is not None and len(all_lines) >= max_samples:
                break

    total = len(all_lines)
    print(f"模型: {model}")
    print(f"样本总数: {total}")
    print(f"线程数: 20\n")
    print("-" * 90)

    results = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_idx = {
            executor.submit(evaluate_one, idx, line): idx
            for idx, line in all_lines
        }

        for future in as_completed(future_to_idx):
            result = future.result()
            results.append(result)
            completed_count += 1
            print_result(result)

    # --- 汇总 ---
    print("\n" + "=" * 90)
    print("评测汇总")
    print("=" * 90)

    correct = sum(1 for r in results if r['is_correct'])
    errors = sum(1 for r in results if r['error'])
    no_judgment = sum(1 for r in results if r['judgment'] is None and not r['error'])
    safe_total = sum(1 for r in results if r['expected'] == 'safe')
    safe_correct = sum(1 for r in results if r['expected'] == 'safe' and r['is_correct'])
    unsafe_total = sum(1 for r in results if r['expected'] == 'unsafe')
    unsafe_correct = sum(1 for r in results if r['expected'] == 'unsafe' and r['is_correct'])
    total_duration = sum(r['duration'] for r in results)

    print(f"总样本数:     {total}")
    print(f"正确数:       {correct}")
    print(f"错误数:       {total - correct - no_judgment - errors}")
    print(f"无判断:       {no_judgment}")
    print(f"API 错误:     {errors}")
    print()
    if total > 0:
        print(f"整体准确率:   {correct / total * 100:.2f}%")
    if safe_total > 0:
        print(f"safe 类:      {safe_correct}/{safe_total} = {safe_correct / safe_total * 100:.2f}%")
    if unsafe_total > 0:
        print(f"unsafe 类:    {unsafe_correct}/{unsafe_total} = {unsafe_correct / unsafe_total * 100:.2f}%")
    print(f"总耗时:       {total_duration:.2f}s")
    print(f"平均耗时:     {total_duration / total:.2f}s/条" if total > 0 else "")

    # 错误详情
    error_details = [r for r in results if not r['is_correct'] and not r['error']]
    if error_details:
        print(f"\n--- 判断错误详情（共 {len(error_details)} 条）---")
        for r in error_details[:20]:
            resp_short = r['response_text'].replace('\n', ' ').strip()[:120]
            print(f"  #[{r['line']}] 期望={r['expected']}, 模型={r['judgment']}")
            print(f"    user: {r['user_preview']}...")
            print(f"    resp: {resp_short}")
            print()
        if len(error_details) > 20:
            print(f"  ... 还有 {len(error_details) - 20} 条未显示")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='安全提示词并发评测脚本')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大评测样本数（默认全部）')
    parser.add_argument('--label', type=str, default=None,
                        choices=['safe', 'unsafe'],
                        help='只评测指定标签的样本')
    args = parser.parse_args()

    model = get_model()
    evaluate(max_samples=args.max_samples, filter_label=args.label)
