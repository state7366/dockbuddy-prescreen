# -*- coding: utf-8 -*-
"""prescreen.asr —— 祖先序列重建(ASR) 生成器(2026-10-11 新增)
===================================================================================
定位: 候选突变生成器之一 —— 从 MSA 重建祖先序列, diff WT 得突变清单(热稳定 +
更宽底物谱候选)。纯 Python(仅依赖 biopython 的 Bio.Phylo, 项目已有), 本机可跑。

算法(诚实标注, 与 PAML 最大似然 ASR 的区别):
  - 本模块用 **Fitch 简约法(parsimony) + 距离法建树(UPGMA/NJ)**, 是 ASR 的
    **快速近似**, 不涉核苷酸/氨基酸替换模型与分支长度最大化似然。
  - PAML codeml(最大似然, ML-ASR)更精确但需外部二进制 + 更重流程; 本模块不依赖
    外部二进制, 可离线即用。结论级别: parsimony ASR 用于「候选生成」足够, 用于
    论文的祖先序列断言需换 ML-ASR(PAML)复核。

流程:
  1. 读 MSA(a3m/fasta) → 对齐序列 + reference。
  2. 建树: p-distance → UPGMA/NJ(Bio.Phylo)。
  3. Fitch 后序(每列状态集) + 前序(定状态)重建指定内部节点的祖先序列。
  4. 祖先序列 diff WT → 突变清单(与 MutationSet schema 兼容)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def p_distance(a: str, b: str) -> float:
    """两对齐序列的 p-distance(忽略 gap 列; 无有效列时返回 1.0)。"""
    n = 0
    diff = 0
    for x, y in zip(a, b):
        if x == "-" or y == "-":
            continue
        n += 1
        if x != y:
            diff += 1
    return diff / n if n else 1.0


def build_tree(seqs: Sequence[str], method: str = "upgma"):
    """从对齐序列建树(p-distance + UPGMA/NJ), 返回 (Bio.Phylo.Tree, leaf_names)。

    leaf_names 与 seqs 顺序一致, 命名 s0..sN-1(树内叶子 .name 对应)。
    """
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

    n = len(seqs)
    names = [f"s{i}" for i in range(n)]
    # Bio.Phylo DistanceMatrix 下三角格式: matrix[i] 有 i+1 个元素(含对角线 0)
    lower: List[List[float]] = []
    for i in range(n):
        lower.append([p_distance(seqs[i], seqs[j]) for j in range(i + 1)])
    dm = DistanceMatrix(names, lower)
    ctor = DistanceTreeConstructor()
    tree = ctor.upgma(dm) if method == "upgma" else ctor.nj(dm)
    return tree, names


def _postorder_sets(clade, leaf_seq: Dict[str, str], L: int) -> List[set]:
    """Fitch 后序: 返回 clade 每个位点的可能状态集(小写/gap 已归一)。"""
    if clade.is_terminal():
        seq = leaf_seq.get(clade.name, "")
        out = []
        for i in range(L):
            aa = seq[i].upper() if i < len(seq) else "-"
            out.append({aa} if aa in _AA_SET else set())
        return out
    child_sets = [_postorder_sets(c, leaf_seq, L) for c in clade.clades]
    out = []
    for i in range(L):
        cols = [cs[i] for cs in child_sets]
        inter = set.intersection(*cols) if cols else set()
        if inter:
            out.append(inter)
        else:
            union = set.union(*cols) if cols else set()
            out.append(union)
    return out


def _leaf_freq(seqs: Sequence[str], L: int) -> List[Dict[str, int]]:
    """每个位点的叶子残基频率 [{aa: count}, ...](忽略 gap)。用于 Fitch 加权 tie-break。"""
    freq: List[Dict[str, int]] = []
    for i in range(L):
        d: Dict[str, int] = {}
        for s in seqs:
            aa = s[i].upper() if i < len(s) else "-"
            if aa in _AA_SET:
                d[aa] = d.get(aa, 0) + 1
        freq.append(d)
    return freq


def _assign_states(clade, parent_states: Optional[List[str]], sets_map: Dict,
                   result: Dict, L: int, leaf_freq: List[Dict[str, int]]) -> List[str]:
    """Fitch 前序: 确定每个 clade 每位点的实际状态, 写入 result[clade_name] = seq。

    tie-break 用叶子频率加权(选出现最多的残基), 避免字典序偏置(如根 {F,L} 应选 L
    而非 F)。
    """
    if parent_states is None:  # 根: 从可能集选叶子频率最高者
        states = [max(s, key=lambda aa: leaf_freq[i].get(aa, 0)) if s else "-"
                  for i, s in enumerate(sets_map[id(clade)])]
    else:
        states = []
        own = sets_map[id(clade)]
        for i in range(L):
            if parent_states[i] in own[i]:
                states.append(parent_states[i])
            elif own[i]:
                states.append(max(own[i], key=lambda aa: leaf_freq[i].get(aa, 0)))
            else:
                states.append("-")
    result[clade.name or f"node{id(clade)}"] = "".join(states)
    for c in clade.clades:
        _assign_states(c, states, sets_map, result, L, leaf_freq)
    return states


def ancestral_sequences(seqs: Sequence[str], tree, leaf_names: Sequence[str],
                        node: str = "") -> Dict[str, str]:
    """Fitch 重建所有内部节点的祖先序列, 返回 {clade_name: sequence}。

    node 非空时仅返回该节点; 缺省返回根节点。
    """
    leaf_seq = dict(zip(leaf_names, seqs))
    L = len(seqs[0]) if seqs else 0
    if not L:
        return {}

    # 后序: 记录每个 clade 的位点状态集(以 id(clade) 为键)
    sets_map: Dict[int, List[set]] = {}

    def _collect(clade):
        sets_map[id(clade)] = _postorder_sets(clade, leaf_seq, L)
        for c in clade.clades:
            _collect(c)

    _collect(tree.root)
    result: Dict[str, str] = {}
    # 前序: 根 parent=None; tie-break 按叶子频率
    _assign_states(tree.root, None, sets_map, result, L, _leaf_freq(seqs, L))

    if node:
        # 按名称/序号匹配内部节点
        for name, seq in result.items():
            if name == node or (name.startswith("node") and name[4:] == node):
                return {name: seq}
        raise ValueError(f"未找到祖先节点 {node!r}(可用节点: {sorted(result)})")
    # 缺省: 返回根(第一个赋值的即根)
    return {next(iter(result)): next(iter(result.values()))}


def diff_to_wt(wt_seq: str, ancestral_seq: str) -> List[Tuple[int, str, str]]:
    """祖先序列 diff WT, 返回 [(site 1-based, wt_aa, anc_aa), ...](忽略 gap 与越界)。"""
    out = []
    L = min(len(wt_seq), len(ancestral_seq))
    for i in range(L):
        wt = wt_seq[i].upper()
        anc = ancestral_seq[i].upper()
        if wt in _AA_SET and anc in _AA_SET and wt != anc:
            out.append((i + 1, wt, anc))
    return out


def asr_mutations(msa_seqs: Sequence[str], reference: str, wt_seq: str, *,
                  method: str = "upgma", node: str = "") -> Dict[str, Any]:
    """端到端: MSA → 建树 → Fitch → 祖先序列 → diff WT。

    返回 {tree, method, node, ancestral_sequence, mutations: [(site, wt, anc), ...]}。
    """
    tree, names = build_tree(msa_seqs, method)
    anc = ancestral_sequences(msa_seqs, tree, names, node=node)
    node_name, anc_seq = next(iter(anc.items()))
    diffs = diff_to_wt(wt_seq or reference, anc_seq)
    return {
        "method": f"fitch+{method}",
        "node": node_name,
        "ancestral_sequence": anc_seq,
        "mutations": diffs,
    }


__all__ = ["build_tree", "ancestral_sequences", "diff_to_wt", "asr_mutations",
           "p_distance"]
