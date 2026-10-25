# -*- coding: utf-8 -*-
"""prescreen.sequence_scan —— 序列进化类候选突变生成器(2026-10-11 新增)
===================================================================================
定位: 候选突变生成器之一组 —— 从 MSA/序列信息提取「值得突变的位点 + 建议替换」。

  pssm     PSSM/Shannon 熵保守性 → 高可变位点 → 家族见过的其他残基(经典)
  plm-scan ESM-1v/ESM-2 零样本全位点饱和扫描 → top-N(前沿, 复用已有打分)

纯 Python(pssm) + 复用 scoring.esm/esm2(plm-scan), 本机可跑。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def position_freqs(seqs: Sequence[str], L: int) -> List[Dict[str, float]]:
    """每位点氨基酸频率(忽略 gap), 返回 [{aa: freq}, ...](freq 归一化)。"""
    out: List[Dict[str, float]] = []
    for i in range(L):
        counts: Dict[str, int] = {}
        for s in seqs:
            c = s[i].upper() if i < len(s) else "-"
            if c in _AA_SET:
                counts[c] = counts.get(c, 0) + 1
        total = sum(counts.values()) or 1
        out.append({aa: n / total for aa, n in counts.items()})
    return out


def shannon_entropy(freqs: Dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in freqs.values() if p > 0)


def pssm_variable_sites(seqs: Sequence[str], reference: str, *,
                        entropy_thr: float = 0.5) -> List[Tuple[int, float]]:
    """高可变位点(Shannon 熵 > entropy_thr), 返回 [(site 1-based, entropy)]。"""
    L = len(reference)
    freqs = position_freqs(seqs, L)
    out = []
    for i, f in enumerate(freqs):
        ent = shannon_entropy(f)
        if ent > entropy_thr:
            out.append((i + 1, ent))
    return sorted(out, key=lambda x: -x[1])


def pssm_mutations(seqs: Sequence[str], reference: str, wt_seq: str, *,
                   entropy_thr: float = 0.5, min_freq: float = 0.05,
                   max_sites: int = 50) -> List[Tuple[int, str, str, str]]:
    """高可变位点 → 家族见过(频率≥min_freq)且非 WT 的残基。

    返回 [(site, wt_aa, alt_aa, reason), ...]。
    """
    L = min(len(reference), len(wt_seq))
    freqs = position_freqs(seqs, L)
    out: List[Tuple[int, str, str, str]] = []
    for i, f in enumerate(freqs):
        site = i + 1
        ent = shannon_entropy(f)
        if ent <= entropy_thr:
            continue
        wt = wt_seq[i]
        for aa, fr in sorted(f.items(), key=lambda kv: -kv[1]):
            if aa != wt and fr >= min_freq:
                out.append((site, wt, aa, f"可变位点熵={ent:.2f}, 家族 {aa} {fr:.0%}"))
    return out[:max_sites]


def plm_scan(seq: str, sites: Sequence[int], *, model_name: str = "esm2",
             engine: str = "esm2", device: str = "cpu",
             top_n: int = 20, exclude: Sequence[str] = ()) -> List[Tuple[str, float]]:
    """ESM-1v/ESM-2 零样本全位点饱和扫描, 返回 [(variant_id, score), ...] 按分降序。

    engine: 'esm1v'(5 种子平均) / 'esm2'(单模型)。exclude: 排除的氨基酸。
    """
    from .mutlib import saturation_library

    excl = list(exclude)
    ms = saturation_library(seq, list(sites), source="plm-scan", exclude=excl)
    variants = [{"site": v.site, "from": v.from_aa, "to": v.to_aa}
                for v in ms.variants]
    if engine == "esm1v":
        from dockbuddy_scoring.esm import esm1v_scores
        scores = esm1v_scores(seq, variants, device=device)
    else:
        from dockbuddy_scoring.esm2 import esm2_scores
        scores = esm2_scores(seq, variants, model_name=model_name, device=device)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]


__all__ = ["position_freqs", "shannon_entropy", "pssm_variable_sites",
           "pssm_mutations", "plm_scan"]
