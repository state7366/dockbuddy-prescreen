# -*- coding: utf-8 -*-
"""prescreen.structure_scan —— 结构理性设计规则生成器(2026-10-11 新增)
===================================================================================
定位: 候选突变生成器之一组 —— 用**经典结构理性设计规则**从 PDB 提取「值得突变的
位点 + 建议替换」, 输出 MutationSet 兼容的突变清单。全部纯 Python + biopython,
本机可跑, 不涉能量函数或 ML 推理。

包含 8 条经典规则(每条一个生成函数):
  bfactor   高 B-factor(柔性)位点 → 刚性化替换(增刚性/热稳定)
  disulfide Cα-Cα 距离近的残基对 → 半胱氨酸(引入二硫键, 增热稳定)
  saltbridge 带相反电荷的残基对 → 强化盐桥(增稳定)
  proline   loop 区甘氨酸 → 脯氨酸(刚性化 loop, 增热稳定)
  surface   溶剂暴露带电残基 → 中性/相反电荷(表面电荷优化)
  hbond     埋藏极性残基 → 优化氢键(增稳定)
  cavity    口袋位点邻近的小侧链 → 大侧链(空腔填充)
  aromatic  底物沟附近残基 → 芳香残基(π-π 锚定, 针对塑料降解底物结合)

诚实边界: 这些是**理性设计启发式规则**(基于结构几何/理化性质), 产出的候选需
下游打分/对接进一步验证; 不预测活性或 ΔΔG。二级结构与溶剂可及性为简化近似
(phi/psi 规则 + Shrake-Rupley SASA), 非 DSSP 精确值。
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
_POS = set("KR")          # 正电荷
_NEG = set("DE")          # 负电荷
_POLAR = set("STNQ")      # 极性(可成氢键)
_SMALL = set("GAS")       # 小侧链(可填充)
_AROMATIC = set("FYW")    # 芳香(π-π 锚定)


def parse_pdb(pdb_path: str, chain: str = "A"):
    """解析 PDB, 返回 {resid(1-based): {aa, ca, bfactor, atoms}}。

    ca = (x,y,z); bfactor = CA 原子 B-factor; atoms = {atom_name: (x,y,z)}。
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    model = struct[0]
    out: Dict[int, Dict[str, Any]] = {}
    target = model[chain] if chain in model else None
    if target is None and chain:
        for ch in model:
            if ch.id == chain:
                target = ch
                break
    if target is None:
        target = next(iter(model))  # 缺省第一条链
    for res in target:
        rid = res.id[1]
        if not isinstance(rid, int):
            continue
        resname = res.get_resname()
        aa = protein_letters_3to1.get(resname, "X")
        if aa not in _AA_SET:
            continue
        atoms: Dict[str, Tuple[float, float, float]] = {}
        ca = None
        bf = 0.0
        for at in res:
            name = at.get_name()
            coord = tuple(float(c) for c in at.get_coord())
            atoms[name] = coord
            if name == "CA":
                ca = coord
                bf = float(at.get_bfactor())
        if ca is None:
            continue
        out[rid] = {"aa": aa, "ca": ca, "bfactor": bf, "atoms": atoms}
    return out


def sasa_by_residue(pdb_path: str, chain: str = "A") -> Dict[int, float]:
    """Shrake-Rupley 溶剂可及表面积, 返回 {resid: sasa_Å²}。"""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    sr = ShrakeRupley()
    sr.compute(struct, level="R")
    out: Dict[int, float] = {}
    for model in struct:
        for ch in model:
            if chain and ch.id != chain:
                continue
            for res in ch:
                rid = res.id[1]
                if isinstance(rid, int) and res.sasa is not None:
                    out[rid] = float(res.sasa)
    return out


def secondary_structure(pdb_path: str, chain: str = "A") -> Dict[int, str]:
    """phi/psi 简化二级结构: 返回 {resid: 'H'(helix)/'E'(sheet)/'C'(coil)}。

    用 PPBuilder 提取多肽链 + phi/psi 二面角, 按粗略区间分类(非 DSSP 精确值)。
    """
    import math
    from Bio.PDB import PDBParser, PPBuilder

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    model = struct[0]
    target = model[chain] if chain in model else next(iter(model))
    ppb = PPBuilder()
    out: Dict[int, str] = {}
    for poly in ppb.build_peptides(target):
        for res, phipsi in zip(poly, poly.get_phi_psi_list()):
            rid = res.id[1]
            if not isinstance(rid, int):
                continue
            phi, psi = phipsi
            if phi is None or psi is None:
                out[rid] = "C"
                continue
            ph = math.degrees(phi)
            ps = math.degrees(psi)
            if -100 <= ph <= -30 and -70 <= ps <= -10:
                out[rid] = "H"
            elif -180 <= ph <= -40 and 80 <= ps <= 180:
                out[rid] = "E"
            else:
                out[rid] = "C"
    return out


def _ca_distance(p1, p2) -> float:
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5


# ---------------------------------------------------------------------------
# 规则生成函数: 每个返回 [(site 1-based, wt_aa, mt_aa, reason), ...]
# ---------------------------------------------------------------------------
def design_bfactor(struct: Dict, seq: str, *, k: float = 1.0) -> List[Tuple]:
    """高 B-factor(柔性)位点 → 刚性化替换(Gly→Ala, 柔性→Pro/大侧链)。

    AF 预测结构的 B-factor 列常为 0(无实验柔性信息), 此时抛 ValueError 提示需
    实验结构或含 pLDDT 的 PDB。
    """
    bf = [(r, d["bfactor"]) for r, d in struct.items()]
    if not bf:
        return []
    mean = sum(v for _, v in bf) / len(bf)
    var = sum((v - mean) ** 2 for _, v in bf) / len(bf)
    std = var ** 0.5
    if std < 1e-6:
        raise ValueError(
            "B-factor 全为同一值(该结构无实验 B-factor; AF 预测结构通常全 0), "
            "B-factor 工程需实验晶体结构或含 B-factor/pLDDT 的 PDB")
    thr = mean + k * std
    out = []
    for r, d in struct.items():
        if d["bfactor"] <= thr or r > len(seq):
            continue
        aa = seq[r - 1]
        # 刚性化规则: Gly→Ala(增刚); 其它柔性残基→Pro(loop 刚化)
        if aa == "G":
            out.append((r, "G", "A", f"B-factor {d['bfactor']:.1f} > {thr:.1f}"))
        elif aa in "ASDNEKT":
            out.append((r, aa, "P", f"B-factor {d['bfactor']:.1f} > {thr:.1f}"))
    return out


def design_disulfide(struct: Dict, seq: str, *, max_dist: float = 7.5,
                     min_seq_dist: int = 4) -> List[Tuple]:
    """Cα-Cα 距离近(<max_dist)且序列远离(≥min_seq_dist)的非半胱氨酸残基对 → Cys。"""
    rids = sorted(struct.keys())
    out = []
    for i, r1 in enumerate(rids):
        for r2 in rids[i + 1:]:
            if r1 > len(seq) or r2 > len(seq):
                continue
            if r2 - r1 < min_seq_dist:
                continue  # 序列邻近残基天然近, 非二硫键候选
            d = _ca_distance(struct[r1]["ca"], struct[r2]["ca"])
            if d > max_dist:
                continue
            a1, a2 = seq[r1 - 1], seq[r2 - 1]
            if a1 != "C" and a2 != "C":
                out.append((r1, a1, "C", f"Cα-Cα {d:.1f}Å 距 {r2}"))
                out.append((r2, a2, "C", f"Cα-Cα {d:.1f}Å 距 {r1}"))
    return out


def design_saltbridge(struct: Dict, seq: str, *, max_dist: float = 8.0,
                      min_seq_dist: int = 2) -> List[Tuple]:
    """带电残基附近的中性残基 → 相反电荷(引入新盐桥, 增稳定)。"""
    rids = sorted(struct.keys())
    out = []
    for i, r1 in enumerate(rids):
        if r1 > len(seq):
            continue
        a1 = seq[r1 - 1]
        if a1 not in _POS and a1 not in _NEG:
            continue
        want = "E" if a1 in _POS else "K"  # 正电荷旁 → 负电荷; 负电荷旁 → 正电荷
        for r2 in rids:
            if r2 == r1 or r2 > len(seq):
                continue
            if abs(r2 - r1) < min_seq_dist:
                continue
            d = _ca_distance(struct[r1]["ca"], struct[r2]["ca"])
            if d > max_dist:
                continue
            a2 = seq[r2 - 1]
            if a2 in _POS or a2 in _NEG or a2 == "P":
                continue
            out.append((r2, a2, want, f"距带电残基 {r1}({a1}) {d:.1f}Å 引入盐桥"))
    return out


def design_proline(struct: Dict, ss: Dict, seq: str) -> List[Tuple]:
    """loop/coil 区甘氨酸 → 脯氨酸(刚性化 loop)。"""
    out = []
    for r, d in struct.items():
        if r > len(seq):
            continue
        if ss.get(r, "C") != "C":
            continue
        if seq[r - 1] == "G":
            out.append((r, "G", "P", "loop 区 Gly→Pro 刚性化"))
    return out


def design_surface(sasa: Dict, struct: Dict, seq: str, *,
                   thr: float = 60.0) -> List[Tuple]:
    """高溶剂暴露(SASA>thr)的带电残基 → 中性/相反电荷(表面电荷优化)。"""
    out = []
    for r, s in sasa.items():
        if r not in struct or r > len(seq) or s <= thr:
            continue
        aa = seq[r - 1]
        if aa in _NEG:
            out.append((r, aa, "N", f"暴露负电荷 SASA={s:.0f}→中性"))
        elif aa in _POS:
            out.append((r, aa, "Q", f"暴露正电荷 SASA={s:.0f}→中性"))
    return out


def design_hbond(sasa: Dict, struct: Dict, seq: str, *,
                 thr: float = 10.0) -> List[Tuple]:
    """埋藏(SASA<thr)的极性残基 → 优化氢键(Ser→Thr 增侧链氢键能力)。"""
    out = []
    for r, s in sasa.items():
        if r not in struct or r > len(seq) or s >= thr:
            continue
        aa = seq[r - 1]
        if aa in _POLAR:
            # 埋藏极性残基若未饱和氢键, 换成侧链更长的同极性残基
            mapping = {"S": "T", "T": "N", "N": "Q", "Q": "H"}
            if aa in mapping:
                out.append((r, aa, mapping[aa],
                            f"埋藏极性 SASA={s:.0f} 优化氢键"))
    return out


def design_cavity(struct: Dict, seq: str, pocket_sites: Sequence[int],
                  *, radius: float = 6.0) -> List[Tuple]:
    """口袋位点空间邻近(<radius Å)的小侧链 → 大侧链(空腔填充, 增稳定/底物接触)。"""
    pocket = [int(s) for s in pocket_sites if int(s) in struct]
    if not pocket:
        return []
    out = []
    for r, d in struct.items():
        if r > len(seq) or r in pocket:
            continue
        if any(_ca_distance(d["ca"], struct[p]["ca"]) <= radius for p in pocket):
            aa = seq[r - 1]
            mapping = {"G": "A", "A": "L", "S": "T", "V": "I"}
            if aa in mapping:
                out.append((r, aa, mapping[aa], "口袋邻近小侧链→大侧链填充"))
    return out


def design_aromatic(struct: Dict, seq: str, pocket_sites: Sequence[int],
                    *, radius: float = 6.0) -> List[Tuple]:
    """底物沟空间邻近(<radius Å)的非芳香残基 → 芳香残基(F, π-π 锚定)。"""
    pocket = [int(s) for s in pocket_sites if int(s) in struct]
    if not pocket:
        return []
    out = []
    for r, d in struct.items():
        if r > len(seq) or r in pocket:
            continue
        if any(_ca_distance(d["ca"], struct[p]["ca"]) <= radius for p in pocket):
            aa = seq[r - 1]
            if aa not in _AROMATIC and aa not in set("PCG"):
                out.append((r, aa, "F", "底物沟邻近→芳香锚定"))
    return out


__all__ = ["parse_pdb", "sasa_by_residue", "secondary_structure",
           "design_bfactor", "design_disulfide", "design_saltbridge",
           "design_proline", "design_surface", "design_hbond",
           "design_cavity", "design_aromatic"]
