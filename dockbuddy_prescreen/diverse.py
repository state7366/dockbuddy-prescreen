# -*- coding: utf-8 -*-
"""
libdiverse.py —— 多样性聚类采样(预筛管线 MVP, 2026-10-08 新增)
===================================================================================
定位: 从「已过布尔闸」的候选里挑出 n 个**彼此差异最大**的子集, 避免把有限的湿实验
名额浪费在冗余候选上(如同一口袋位点塞满 19 种氨基酸的饱和库, 全测既贵又重复)。

MVP 特征空间(用户拍板: 简化聚类, RMSD 推迟):
  - **site**(位点): 不同位点 → 越多样。
  - **aa_type**(氨基酸类型): 5 类 {aliphatic/aromatic/polar/charged/other}。
  用贪心最远点采样(farthest-point sampling)在 one-hot 特征向量上做最大最小化,
  逐点挑「与已选集最近邻距离最大」者, 确定性平局按 id 排序。

设计决策(记录备查):
  - **不做单一综合总分排序**: 本模块只做多样性采样, 与打分正交(打分已由布尔闸消费)。
  - **确定性**: 候选先按 (id, site, to) 排序, 首挑取排序首项(或显式 seed_id),
    平局取更小下标 —— 同一输入必给同一输出(publication-grade 可复现)。
  - **RMSD/结构聚类推迟到全量**: 结构相近但位点/类型不同的候选在 MVP 不区分,
    那是全量阶段接 RMSD 特征后的工作。

aa_type 分类(标准 5 类):
  aliphatic: G A V L I M P ; aromatic: F Y W ; polar: S T N Q C ;
  charged: D E K R H ; 其余 -> other。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

LOG = logging.getLogger("dockpipeline")

_AA_TYPE: Dict[str, str] = {
    "G": "aliphatic", "A": "aliphatic", "V": "aliphatic", "L": "aliphatic",
    "I": "aliphatic", "M": "aliphatic", "P": "aliphatic",
    "F": "aromatic", "Y": "aromatic", "W": "aromatic",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar", "C": "polar",
    "D": "charged", "E": "charged", "K": "charged", "R": "charged", "H": "charged",
}
TYPE_ORDER: Tuple[str, ...] = ("aliphatic", "aromatic", "polar", "charged", "other")
DEFAULT_FEATURES: Tuple[str, ...] = ("site", "aa_type")


def aa_type(aa: str) -> str:
    """单字母 → 五类之一(未知 -> other)。"""
    return _AA_TYPE.get(str(aa).strip().upper(), "other")


def _type_index(t: str) -> int:
    return TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER) - 1


def _feature_vec(cand: Dict[str, Any], site_idx: Dict[int, int],
                 features: Sequence[str], site_weight: float = 2.0) -> List[float]:
    """one-hot 特征向量 = site 维度(× site_weight) + aa_type 维度。

    site_weight>1 的意义(实测修正, 2026-10-08): 不加权时「同一位点换类型」与
    「新位点同类型」的欧氏距离同为 √2, 贪心平局按排序下标取, 会倾向把名额花在
    **同一位点的多个氨基酸**上、漏掉尚未覆盖的位点(违背「不同位点→越多样」的
    意图)。给 site 维度乘 >1 的权重后, 覆盖新位点(距离 2√2·w)恒大于同位点换
    类型(距离 √2), 贪心自然先铺满位点再回填类型。
    """
    vec: List[float] = []
    if "site" in features:
        onehot = [0.0] * len(site_idx)
        onehot[site_idx[int(cand["site"])]] = site_weight
        vec.extend(onehot)
    if "aa_type" in features:
        onehot = [0.0] * len(TYPE_ORDER)
        onehot[_type_index(aa_type(cand.get("to", "")))] = 1.0
        vec.extend(onehot)
    return vec


def _dist(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def select_diverse(candidates: Sequence[Dict[str, Any]],
                   n_select: int,
                   *,
                   features: Sequence[str] = DEFAULT_FEATURES,
                   id_key: str = "id",
                   seed_id: str = "",
                   site_weight: float = 2.0) -> Dict[str, Any]:
    """贪心最远点采样, 从 candidates 挑 n_select 个多样性最大化子集。

    candidates 每条至少含 {site:int, to:单字母}; 可选 id(缺省用 key/variant_id)。
    返回 {"selected": [候选 dict], "order": [id...], "coverage": {...}}。
    site_weight 控制「覆盖新位点 vs 同位点换类型」的优先级(见 _feature_vec)。
    n_select < 1 表示全选(排序后全部返回, 不做裁剪)。
    """
    if not candidates:
        return {"selected": [], "order": [], "coverage": {}}
    cands = list(candidates)
    cands.sort(key=lambda c: (str(c.get(id_key) or c.get("key") or c.get("variant_id") or ""),
                              int(c.get("site", 0)), str(c.get("to", ""))))
    sites = sorted({int(c["site"]) for c in cands})
    site_idx = {s: i for i, s in enumerate(sites)}
    vecs = [_feature_vec(c, site_idx, features, site_weight) for c in cands]

    target = len(cands) if n_select < 1 else min(int(n_select), len(cands))

    selected: List[int] = []
    if seed_id:
        for i, c in enumerate(cands):
            if (c.get(id_key) or c.get("variant_id") or c.get("key")) == seed_id:
                selected.append(i)
                break
    if not selected:
        selected.append(0)                       # 确定性首挑: 排序首项
    pool = [i for i in range(len(cands)) if i not in selected]

    while len(selected) < target and pool:
        best, best_d = -1, -1.0
        for i in pool:
            d = min(_dist(vecs[i], vecs[j]) for j in selected)
            if best < 0 or d > best_d or (d == best_d and i < best):
                best, best_d = i, d
        if best < 0:
            break
        selected.append(best)
        pool.remove(best)

    picked = [cands[i] for i in selected]
    order = [c.get(id_key) or c.get("key") or c.get("variant_id") or "" for c in picked]
    coverage = {
        "n_pool": len(cands),
        "n_selected": len(picked),
        "sites_covered": sorted({int(c["site"]) for c in picked}),
        "types_selected": sorted({aa_type(c.get("to", "")) for c in picked}),
    }
    return {"selected": picked, "order": order, "coverage": coverage}


__all__ = ["aa_type", "select_diverse", "DEFAULT_FEATURES", "TYPE_ORDER"]
