# -*- coding: utf-8 -*-
"""
libmutlib.py —— 突变库生成与标准化(预筛管线 MVP, 2026-10-07 新增)
===================================================================================
定位: 预筛管线第一阶段「突变库生成」的实现。它只负责**枚举候选变体并落地为
标准化 mutationset.json**, 不做任何打分/对接/决策(那是 libgate / libdiverse /
orchestrator 的职责)。本模块是**纯 Python、零科学计算依赖** —— 仅复用
libmutate 的突变解析与氨基酸表(parse_mutation / AA1_TO_3 / AA3_TO_1), 不 import
openmm / numpy, 因此业务逻辑的单元测试在 CI 无分子模拟环境下可绿(满足
「单测分层: 业务逻辑零外部依赖」)。

标准化 schema (mutationset.json):
    {
      "schema_version": "1.0",
      "enzyme_id": "PBATase",
      "template_sequence": "MSEQN...",
      "source": "saturation",          # 生成方式, 仅作记录
      "created": "2026-10-07",         # 可选, 空字符串表示未填
      "positions": [                   # 派生展示: 按位点聚合, 不入库(由 variants 推导)
        {"site": 205, "wt": "F", "variants": ["G","A","L"], "source": "...", "n_variants": 3}
      ],
      "variants": [                    # 规范事实源(canonical): 去重后的变体列表
        {"id": "F205G", "site": 205, "from": "F", "to": "G", "source": "saturation"}
      ]
    }

设计决策(记录备查):
  - **WT 身份硬校验**: 任何生成路径都强制 variant.from 与 template_sequence[site-1]
    一致, 不一致直接 raise(与 libmutate 的「WT 残基身份逐位点硬校验」同口径, 防止
    编号错位静默错突)。
  - **1-letter 对外、3-letter 内部只经 libmutate 转换**: schema 用 F205G 这种人类
    可读的 1-letter 表示; 内部 Variant.from_aa/to_aa 也是 1-letter; 仅在吃
    libmutate.parse_mutation 的 3-letter 结果时经 AA3_TO_1 归一。
  - **positions 是派生量**: to_dict 时由 variants 聚合生成, from_dict 时丢弃 positions
    (永远以 variants 为准重建), 杜绝两份真相漂移。
  - **去重键 (site, from_aa, to_aa, chain)**: 同链同残基同目标只留一份, 多源合并时
    保留首次出现(来源可不同, 但候选本身唯一)。
  - **不做自我突变(WT→WT)**: 生成时跳过 from==to 的条目(那是 libmutate 的对称对照
    概念, 不属于「候选变体库」; 若确需对称对照由下游显式构造)。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

LOG = logging.getLogger("dockpipeline")

# 复用 libmutate 的解析与氨基酸表: 模块级零重依赖(parse_mutation 只用到正则,
# AA1_TO_3/AA3_TO_1 是纯 dict), 不会触发 openmm/numpy 的导入。
from dockbuddy_mutate.mutate import AA1_TO_3, AA3_TO_1, parse_mutation  # noqa: E402

SCHEMA_VERSION = "1.0"
# 标准 20 种氨基酸单字母(遍历饱和突变用); 顺序固定保证可复现。
_STD_AA = list(AA1_TO_3.keys())


# ----------------------------------------------------------------------------
# 单个变体
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """一个单点突变候选。site 为 1-based 残基号; from_aa/to_aa 为 1-letter。

    JSON 对外字段用 `from`/`to`(与计划 schema 一致), 内部以 from_aa/to_aa 规避
    Python 关键字。chain 为空表示沿用默认链。
    """

    site: int
    from_aa: str
    to_aa: str
    source: str = ""
    chain: str = ""

    @property
    def id(self) -> str:
        """人类可读标识, 与 libmutate.MutSpec.label 同口径: 如 'F205G'。"""
        return f"{self.from_aa}{self.site}{self.to_aa}"

    @property
    def label(self) -> str:
        return self.id

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "site": int(self.site),
            "from": self.from_aa,
            "to": self.to_aa,
            "source": self.source,
        }
        if self.chain:
            d["chain"] = self.chain
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Variant":
        return cls(
            site=int(d["site"]),
            from_aa=str(d["from"]).strip().upper(),
            to_aa=str(d["to"]).strip().upper(),
            source=str(d.get("source", "")),
            chain=str(d.get("chain", "")),
        )


# ----------------------------------------------------------------------------
# 突变库(规范事实源 = variants 列表)
# ----------------------------------------------------------------------------
@dataclass
class MutationSet:
    enzyme_id: str
    template_sequence: str
    variants: List[Variant] = field(default_factory=list)
    source: str = ""
    created: str = ""
    schema_version: str = SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.variants)

    def __iter__(self):
        return iter(self.variants)

    # -- 派生: 按位点聚合(仅展示, 不存储) -------------------------------
    @property
    def positions(self) -> List[Dict[str, Any]]:
        by_site: Dict[int, List[Variant]] = {}
        order: List[int] = []
        for v in self.variants:
            if v.site not in by_site:
                by_site[v.site] = []
                order.append(v.site)
            by_site[v.site].append(v)
        out: List[Dict[str, Any]] = []
        for site in order:
            vs = by_site[site]
            out.append({
                "site": site,
                "wt": vs[0].from_aa,
                "variants": sorted({v.to_aa for v in vs}),
                "source": vs[0].source,
                "n_variants": len(vs),
            })
        return out

    # -- 序列化 ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enzyme_id": self.enzyme_id,
            "template_sequence": self.template_sequence,
            "source": self.source,
            "created": self.created,
            "positions": self.positions,
            "variants": [v.to_dict() for v in self.variants],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MutationSet":
        vs = [Variant.from_dict(v) for v in d.get("variants", [])]
        return cls(
            enzyme_id=str(d.get("enzyme_id", "")),
            template_sequence=str(d.get("template_sequence", "")),
            variants=vs,
            source=str(d.get("source", "")),
            created=str(d.get("created", "")),
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        )

    def write_json(self, path: str) -> str:
        """落盘 mutationset.json(原子性: 先写临时文件再 rename)。返回绝对路径。"""
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    @classmethod
    def read_json(cls, path: str) -> "MutationSet":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    # -- 操作 -----------------------------------------------------------
    def dedup(self) -> "MutationSet":
        """按 (site, from_aa, to_aa, chain) 去重, 保留首次出现。返回新实例。"""
        seen: Dict[Tuple[int, str, str, str], bool] = {}
        uniq: List[Variant] = []
        for v in self.variants:
            key = (v.site, v.from_aa, v.to_aa, v.chain)
            if key in seen:
                continue
            seen[key] = True
            uniq.append(v)
        return MutationSet(
            enzyme_id=self.enzyme_id,
            template_sequence=self.template_sequence,
            variants=uniq,
            source=self.source,
            created=self.created,
            schema_version=self.schema_version,
        )

    def validate_wrt_template(self) -> List[str]:
        """返回所有不一致的错误字符串(空列表 = 全部一致)。

        检查两类: (1) 位点越界; (2) variant.from 与模板实际残基不符。"""
        errs: List[str] = []
        seq = self.template_sequence
        for v in self.variants:
            if v.site < 1 or v.site > len(seq):
                errs.append(f"{v.id}: 位点 {v.site} 超出模板长度 {len(seq)}")
                continue
            actual = seq[v.site - 1].upper()
            if actual != v.from_aa:
                errs.append(f"{v.id}: 起点 {v.from_aa} 与模板实际残基 {actual} 不符")
        return errs


# ----------------------------------------------------------------------------
# 底层工具
# ----------------------------------------------------------------------------
def _aa_at(seq: str, site: int) -> str:
    """返回模板上 1-based 位点的 1-letter 残基; 越界或非标准字符均抛错。"""
    if site < 1 or site > len(seq):
        raise IndexError(f"位点 {site} 超出模板长度 {len(seq)}")
    aa = seq[site - 1].upper()
    if aa not in AA1_TO_3:
        raise ValueError(f"模板位点 {site} 的残基 '{aa}' 非标准氨基酸, 无法构建突变库")
    return aa


# ----------------------------------------------------------------------------
# 生成路径
# ----------------------------------------------------------------------------
def saturation_library(template_seq: str,
                       sites: Sequence[int],
                       *,
                       enzyme_id: str = "",
                       source: str = "saturation",
                       skip_wt: bool = True,
                       exclude: Sequence[str] = ()) -> MutationSet:
    """对给定位点做**饱和突变**: 每个位点枚举除 WT 外的所有标准氨基酸。

    - skip_wt=True: 跳过 from==to(即 WT 本身)。
    - exclude: 额外排除的 1-letter(如 'P','C' 等结构敏感残基)。
    返回 MutationSet(未去重, 但饱和本身不会重复)。
    """
    excl = {str(a).upper() for a in exclude}
    variants: List[Variant] = []
    for site in sites:
        wt = _aa_at(template_seq, site)
        for aa in _STD_AA:
            if skip_wt and aa == wt:
                continue
            if aa in excl:
                continue
            variants.append(Variant(site=site, from_aa=wt, to_aa=aa, source=source))
    return MutationSet(
        enzyme_id=enzyme_id, template_sequence=template_seq,
        variants=variants, source=source)


def from_position_spec(template_seq: str,
                       positions_spec: Sequence[Dict[str, Any]],
                       *,
                       enzyme_id: str = "",
                       source: str = "manual") -> MutationSet:
    """按位点规格构造突变库。

    positions_spec 每条: {"site": int, "variants": [aa...], "wt"?: "F", "source"?: "..."}.
    - wt 可选: 省略则按模板推导; 若显式给出且与模板不符 → ValueError(硬校验)。
    - variants 中等于 WT 的条目被跳过(自我突变不入库, 记 WARNING)。
    - variants 中出现非标准氨基酸 → ValueError。
    """
    variants: List[Variant] = []
    for ps in positions_spec:
        site = int(ps["site"])
        wt_seq = _aa_at(template_seq, site)
        declared = str(ps.get("wt", "")).strip().upper()
        if declared:
            if declared != wt_seq:
                raise ValueError(
                    f"位点 {site} WT 不符: 显式声明 {declared} 但模板实际为 {wt_seq}")
        wt = declared or wt_seq
        ps_src = str(ps.get("source", source))
        for raw in ps["variants"]:
            aa = str(raw).strip().upper()
            if aa == wt:
                LOG.warning("位点 %d 变体 %s 与 WT 相同, 不入库", site, aa)
                continue
            if aa not in AA1_TO_3:
                raise ValueError(f"位点 {site} 变体 '{aa}' 非标准氨基酸")
            variants.append(Variant(site=site, from_aa=wt, to_aa=aa, source=ps_src))
    return MutationSet(
        enzyme_id=enzyme_id, template_sequence=template_seq,
        variants=variants, source=source)


def from_variant_list(template_seq: str,
                      variant_labels: Sequence[str],
                      *,
                      enzyme_id: str = "",
                      source: str = "manual",
                      default_chain: str = "") -> MutationSet:
    """按突变标签列表构造突变库。标签经 libmutate.parse_mutation 解析, 支持
    'F205G' / 'F205G:B' / 'ALA-205-GLY' / 'ALA205GLY'。

    WT 硬校验: parse 得到的 3-letter wt 必须等于模板对应位点的残基, 否则 ValueError。
    """
    variants: List[Variant] = []
    for lab in variant_labels:
        m = parse_mutation(str(lab))
        wt_seq = _aa_at(template_seq, m.resnum)
        if m.wt3 != AA1_TO_3[wt_seq]:
            raise ValueError(
                f"{m.label} WT 不符: 声明 {m.wt3} 但模板位点 {m.resnum} 实际为 {wt_seq}")
        variants.append(Variant(
            site=m.resnum, from_aa=wt_seq, to_aa=AA3_TO_1[m.mut3],
            source=source, chain=(m.chain or default_chain)))
    return MutationSet(
        enzyme_id=enzyme_id, template_sequence=template_seq,
        variants=variants, source=source)


def union(*sets: MutationSet) -> MutationSet:
    """合并多个突变库并去重。模板/酶 ID 以首个非空为准。"""
    sets = [s for s in sets if s is not None]
    if not sets:
        return MutationSet("", "", [])
    merged: List[Variant] = []
    for s in sets:
        merged.extend(s.variants)
    base = next((s for s in sets if s.template_sequence), sets[0])
    out = MutationSet(
        enzyme_id=base.enzyme_id,
        template_sequence=base.template_sequence,
        variants=merged,
        source="union")
    return out.dedup()


__all__ = [
    "SCHEMA_VERSION",
    "Variant",
    "MutationSet",
    "saturation_library",
    "from_position_spec",
    "from_variant_list",
    "union",
]
