# -*- coding: utf-8 -*-
"""用 matplotlib 绘制项目流程图与系统架构图，导出 PNG。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = r"d:\AITest\aitest\Prompt安全检测数据集"

# 配色（标准语义）
C_TERM = "#4FC3F7"   # 起止框（圆角）
C_PROC = "#A5D6A7"   # 处理框（矩形）
C_DECI = "#FFF59D"   # 判断框（菱形）
C_DATA = "#CE93D8"   # 数据/IO（平行四边形）
C_SVC  = "#FFCC80"   # 服务/模型
C_EDGE = "#37474F"


def round_box(ax, cx, cy, w, h, text, fc=C_TERM, fs=14, sub=None):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.16",
                       fc=fc, ec=C_EDGE, lw=1.4, zorder=2)
    ax.add_patch(b)
    if sub:
        ax.text(cx, cy + h * 0.18, text, ha="center", va="center", fontsize=fs,
                fontweight="bold", zorder=3)
        ax.text(cx, cy - h * 0.24, sub, ha="center", va="center", fontsize=fs - 2,
                color="#333", zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold", zorder=3)


def box(ax, cx, cy, w, h, text, fc=C_PROC, fs=14, sub=None):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.02",
                       fc=fc, ec=C_EDGE, lw=1.4, zorder=2)
    ax.add_patch(b)
    if sub:
        ax.text(cx, cy + h * 0.18, text, ha="center", va="center", fontsize=fs,
                fontweight="bold", zorder=3)
        ax.text(cx, cy - h * 0.24, sub, ha="center", va="center", fontsize=fs - 2,
                color="#333", zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold", zorder=3)


def diamond(ax, cx, cy, w, h, text, fs=12):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=C_DECI, ec=C_EDGE, lw=1.4, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", zorder=3)


def para(ax, cx, cy, w, h, text, fs=12):
    s = 0.22
    pts = [(cx - w / 2 + s, cy + h / 2), (cx + w / 2 + s, cy + h / 2),
           (cx + w / 2 - s, cy - h / 2), (cx - w / 2 - s, cy - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=C_DATA, ec=C_EDGE, lw=1.4, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", zorder=3)


def arrow(ax, x1, y1, x2, y2, label=None, lx=0.0, ly=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.6,
                                shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center",
                va="center", fontsize=12, color="#C62828", fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1))


# ========================================================================
# 图一：流程图
# ========================================================================
fig, ax = plt.subplots(figsize=(17, 21))
ax.set_xlim(0, 17)
ax.set_ylim(0, 21)
ax.axis("off")
ax.set_title("Prompt 安全检测项目 —— 端到端流程图", fontsize=20, fontweight="bold", pad=16)

round_box(ax, 8.5, 19.2, 6.2, 1.15, "原始数据", sub="train / val / test.parquet")

box(ax, 8.5, 17.0, 9.0, 1.4, "整理数据（convert_to_jsonl.py）",
    sub="① 只留中文  ② 去掉≤5字的短句  ③ 安全/不安全各抽1.5万条  ④ 转成标准训练格式")

para(ax, 8.5, 15.0, 6.4, 1.05, "整理好的训练/测试数据  sft_*.jsonl")

arrow(ax, 8.5, 18.6, 8.5, 17.85)
arrow(ax, 8.5, 16.25, 8.5, 15.55)

# 分支
arrow(ax, 6.9, 14.45, 3.2, 12.7, label="训练路线", lx=-1.4, ly=0.45)
arrow(ax, 10.1, 14.45, 13.8, 12.7, label="检索路线", lx=1.4, ly=0.45)

box(ax, 3.2, 12.1, 5.6, 1.25, "微调大模型（学会判断安全）", sub="Qwen3-1.7B · LoRA · LLaMA-Factory")
box(ax, 3.2, 10.2, 5.6, 1.15, "启动模型接口服务", sub="llamafactory-cli api")

box(ax, 13.8, 12.1, 5.6, 1.15, "给训练文本生成向量", sub="智谱 embedding-3（文字→数字向量）")
box(ax, 13.8, 10.2, 5.6, 1.15, "建立向量索引库", sub="FAISS · IndexFlatIP（快速找最像的）")

box(ax, 3.2, 8.3, 5.6, 1.1, "模型判断服务", fc=C_SVC, sub="localhost:8000/v1")
para(ax, 13.8, 8.3, 5.0, 1.05, "向量索引文件  faiss_index.bin")

arrow(ax, 3.2, 11.45, 3.2, 10.8)
arrow(ax, 3.2, 9.6, 3.2, 8.9)
arrow(ax, 13.8, 11.5, 13.8, 10.8)
arrow(ax, 13.8, 9.6, 13.8, 8.9)

# 三种评测方案
box(ax, 2.4, 5.4, 3.7, 1.4, "方案A\n只用模型判断", sub="evaluate.py")
box(ax, 8.5, 5.4, 3.7, 1.4, "方案B\n只用向量检索", sub="faiss_evaluate.py")
box(ax, 14.6, 5.4, 3.7, 1.4, "方案C\n先检索、没把握再模型", sub="cascade_evaluate.py")

arrow(ax, 3.2, 8.0, 2.9, 6.2)
arrow(ax, 3.2, 8.0, 13.2, 6.05)
arrow(ax, 13.8, 8.0, 13.2, 6.05)
arrow(ax, 13.8, 8.0, 8.5, 6.2)

# 级联路由细节
diamond(ax, 14.6, 3.1, 3.4, 1.8, "相似度\n够高吗？\ns ≥ T")
arrow(ax, 14.6, 4.7, 14.6, 4.0)

plt.tight_layout()
fig.savefig(f"{BASE}\\流程图.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成: 流程图.png")

# ========================================================================
# 图二：系统架构图
# ========================================================================
fig, ax = plt.subplots(figsize=(17, 11))
ax.set_xlim(0, 17)
ax.set_ylim(0, 11)
ax.axis("off")
ax.set_title("Prompt 安全检测系统 —— 分层架构图", fontsize=20, fontweight="bold", pad=14)


def band(ax, y, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((0.3, y), 16.4, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=C_EDGE, lw=1.4, alpha=0.25, zorder=1))
    ax.text(0.75, y + h * 0.72, title, fontsize=14, fontweight="bold",
            va="center", ha="left", color="#1B5E20", zorder=3)
    ax.text(0.75, y + h * 0.22, sub, fontsize=11, va="center", ha="left",
            color="#33691E", zorder=3)


# 数据层
band(ax, 9.4, 1.5, "① 数据层", "原始素材", "#4FC3F7")
para(ax, 4.6, 10.15, 3.4, 0.95, "原始数据集\ntrain/val/test.parquet", fs=11)
para(ax, 9.0, 10.15, 3.6, 0.95, "训练数据 3万条\nsft_train_zh.jsonl", fs=11)
para(ax, 14.4, 10.15, 3.6, 0.95, "测试数据 3195条\nsft_test_zh.jsonl", fs=11)

# 构建层
band(ax, 7.5, 1.5, "② 准备层", "训练模型 / 建索引", "#A5D6A7")
box(ax, 4.2, 8.25, 4.2, 1.1, "微调大模型", sub="Qwen3-1.7B · LoRA")
box(ax, 9.0, 8.25, 4.2, 1.1, "调用向量化接口", sub="智谱 embedding-3")
box(ax, 14.4, 8.25, 4.2, 1.1, "建立向量索引", sub="FAISS · IndexFlatIP")

# 服务层
band(ax, 5.6, 1.5, "③ 服务层", "对外提供判断", "#FFCC80")
box(ax, 6.4, 6.35, 4.6, 1.1, "模型判断服务", sub="localhost:8000/v1")
box(ax, 13.2, 6.35, 4.6, 1.1, "向量检索服务", sub="faiss 检索 k=1")

# 评测层
band(ax, 3.7, 1.5, "④ 评测层", "三种方案对比", "#CE93D8")
box(ax, 3.9, 4.4, 3.4, 1.1, "方案A 纯微调", sub="evaluate.py")
box(ax, 8.6, 4.4, 3.4, 1.1, "方案B 纯检索", sub="faiss_evaluate.py")
box(ax, 13.6, 4.4, 3.6, 1.1, "方案C 级联", sub="cascade_evaluate.py")

# 级联路由子块
band(ax, 1.8, 1.5, "⑤ 级联如何分工", "cascade_evaluate.py", "#FFF59D")
diamond(ax, 7.6, 2.55, 3.4, 1.5, "相似度\n够高吗？\ns ≥ T")
round_box(ax, 13.0, 2.55, 3.4, 1.0, "计算余弦相似度", sub="IndexFlatIP")

# 纵向连线
arrow(ax, 4.6, 9.65, 4.2, 8.85)
arrow(ax, 9.0, 9.65, 9.0, 8.85)
arrow(ax, 9.0, 9.65, 12.2, 8.85)
arrow(ax, 14.4, 9.65, 13.6, 4.95)
arrow(ax, 4.2, 7.7, 6.4, 7.0)
arrow(ax, 14.4, 7.7, 13.2, 7.0)
arrow(ax, 9.0, 7.7, 11.4, 7.0)

arrow(ax, 6.4, 5.85, 3.9, 5.0)
arrow(ax, 13.2, 5.85, 13.4, 5.0)
arrow(ax, 13.6, 5.85, 13.6, 5.0)
arrow(ax, 8.6, 3.85, 7.6, 3.2)
arrow(ax, 13.6, 3.85, 7.6, 3.2)

# 分支标注
ax.text(13.6, 7.0, "不相似 → 交给模型", fontsize=12, color="#C62828", fontweight="bold")
ax.text(13.6, 3.4, "很相似 → 直接信检索", fontsize=12, color="#1B5E20", fontweight="bold")

plt.tight_layout()
fig.savefig(f"{BASE}\\系统架构图.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成: 系统架构图.png")
print("脚本文件: draw_diagrams.py（可自行修改后再运行）")