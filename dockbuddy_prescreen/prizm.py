# -*- coding: utf-8 -*-
"""
prescreen/prizm.py —— PRIZM 式「低 N 选零样本模型 + 排序」工具(2026-10-23 新增)
====================================================================================
定位: 落地 ima「论文主动发现」新增论文 PRIZM(Bioinformatics Advances 2026,
DOI:10.1093/bioadv/vbag299) 的两阶段工作流 ——
  阶段一(模型选择): 用少量已标注变体(原文约 20 条)评估各候选零样本打分模型与
    标注的相关性, 选出最适合目标蛋白性质的模型;
  阶段二(排序): 用选出的模型对整个变体库打分排序, 输出 top 变体。

DockBuddy 拥有 26+ 打分模型且符号/口径不一(见 docs/模型调用链路手册.md §3.1),
「面对 PBATase 该信哪个模型」是实际痛点; 本模块提供数据驱动的模型选择, 与
SGES(低维子空间, 换坐标) 互补(本模块换模型)。

设计决策(记录备查):
  - 纯 numpy 实现 Spearman 秩相关(优先 scipy, 缺省 numpy 回退), 零硬依赖。
  - 相关性用 Spearman(秩), 与 PRIZM 原文一致(对打分量纲/单调变换鲁棒), 避免
    各模型量纲差异(如 evmutation 的 -10 vs foldx 的 +1.4)污染 Pearson。
  - 输入契约: model_scores = {model: {variant_id: score}}, labels = {variant_id:
    y}; 二者按 variant_id 对齐, 缺标签的变体不进相关性计算。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

__all__ = ["spearman", "rank_models", "select_best_model", "rank_library"]


def _rankdata(a: np.ndarray) -> np.ndarray:
    """秩(平均法处理并列), 等价 scipy.stats.rankdata(a, method='average')。"""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # 并列取平均秩
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman 秩相关系数; 常量输入(零方差)返回 NaN。"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr  # 优先 scipy
        return float(spearmanr(a, b).correlation)
    except Exception:
        pass
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_models(model_scores: Dict[str, Dict[str, float]],
                labels: Dict[str, float]) -> List[Tuple[str, float]]:
    """按 Spearman(打分, 标注) 从高到低排序各模型。

    返回 [(model, rho), ...]; 与标注无交集或相关性为 NaN 的模型排在末尾(rho=NaN)。
    """
    scored: List[Tuple[str, float]] = []
    unranked: List[str] = []
    for model, scores in model_scores.items():
        common = [v for v in labels if v in scores]
        if len(common) < 2:
            unranked.append(model)
            continue
        x = np.array([scores[v] for v in common])
        y = np.array([labels[v] for v in common])
        rho = spearman(x, y)
        if np.isnan(rho):
            unranked.append(model)
        else:
            scored.append((model, rho))
    scored.sort(key=lambda t: t[1], reverse=True)
    # 未评级/NaN 模型置底
    scored += [(m, float("nan")) for m in unranked]
    return scored


def select_best_model(model_scores: Dict[str, Dict[str, float]],
                      labels: Dict[str, float]) -> Tuple[str, float]:
    """选出与标注最相关(最高 Spearman)的零样本模型。返回 (best_model, rho)。"""
    ranked = rank_models(model_scores, labels)
    if not ranked or np.isnan(ranked[0][1]):
        raise ValueError("无任何模型与标注存在有效相关性, 无法选模型。")
    return ranked[0]


def rank_library(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    """按打分从高到低排序变体库(阶段二)。返回 [(variant_id, score), ...]。"""
    items = [(v, s) for v, s in scores.items() if not np.isnan(float(s))]
    items.sort(key=lambda t: t[1], reverse=True)
    return items
