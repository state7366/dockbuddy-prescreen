# -*- coding: utf-8 -*-
"""
layers.py —— 配置与数据的三层分离(protocol / dataset / run)
================================================================================
「怎么算」(方法) 与「算什么」(数据) 的变化率不同, 混在一起导致"换模型跑同一批数据"
要复制整个 manifest。三层分离后:

  Protocol  方法层(可复用/可版本化): 模型选择 + 参数 + 约束 + 阈值
  Dataset   数据层(可复用): 受体 + 配体 + 序列 + 突变库
  Run       组合层: 引用一个 Protocol + 一个 Dataset, 加 run_id/覆盖/产物

三个场景各只改一层:
  同数据换模型  → 换 Protocol(Dataset 不变)
  同模型换数据  → 换 Dataset(Protocol 不变)
  M×N 交叉组合  → run_matrix(protocols, datasets) 笛卡尔积展开

Run.to_manifest() 把 protocol + dataset 合成 orchestrator.run_pipeline 的 manifest;
overrides 支持局部覆盖(如只换 seed)而不复制整个 protocol。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Protocol:
    """方法层: 用哪些模型、什么参数、什么约束。可命名、可版本化、可分享。"""
    name: str
    scoring: Dict[str, Any] = field(default_factory=lambda: {"adapter": "scores_file"})
    constraints: Dict[str, Any] = field(default_factory=dict)
    diversity: Dict[str, Any] = field(default_factory=lambda: {"n_select": 30,
                                                               "features": ["site", "aa_type"]})
    export: Dict[str, Any] = field(default_factory=lambda: {"format": "evolvepro"})
    refine: Optional[Dict[str, Any]] = None
    version: str = "1"


@dataclass
class Dataset:
    """数据层: 算什么。可命名、可复用、可被多个 Run 引用。"""
    name: str
    template_sequence: str = ""
    mutationset: str = ""            # mutationset.json 路径 或 inline dict
    scores_file: str = ""            # 预计算打分(配合 scoring.adapter=scores_file)
    receptor: str = ""               # 受体 PDB(结构依赖的打分/折叠用)
    ligand_smiles: str = ""          # 配体(对接用)


@dataclass
class Run:
    """组合层: 一次运行 = Protocol × Dataset + 覆盖 + 产物。"""
    protocol: Protocol
    dataset: Dataset
    run_id: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)
    out_dir: str = ""

    def to_manifest(self) -> Dict[str, Any]:
        """合成 orchestrator.run_pipeline 的 manifest(protocol + dataset → 单一事实)。"""
        m: Dict[str, Any] = {
            "enzyme_id": self.dataset.name,
            "mutationset": self.dataset.mutationset or {
                "template_sequence": self.dataset.template_sequence},
            "scoring": dict(self.protocol.scoring),
            "constraints": dict(self.protocol.constraints),
            "diversity": dict(self.protocol.diversity),
            "export": dict(self.protocol.export),
        }
        if self.dataset.scores_file:
            m["scores_file"] = self.dataset.scores_file
        if self.dataset.receptor:
            m["receptor"] = self.dataset.receptor
        if self.dataset.ligand_smiles:
            m["ligand_smiles"] = self.dataset.ligand_smiles
        if self.protocol.refine:
            m["refine"] = dict(self.protocol.refine)
        if self.out_dir:
            m["out_dir"] = self.out_dir
        _apply_overrides(m, self.overrides)
        return m

    @property
    def label(self) -> str:
        return self.run_id or f"{self.protocol.name}__{self.dataset.name}"


def _apply_overrides(manifest: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """按点分路径覆盖 manifest 的局部字段(如 'scoring.seed' → manifest['scoring']['seed'])。"""
    for dotted, val in overrides.items():
        parts = dotted.split(".")
        node = manifest
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val


def run_matrix(protocols: Sequence[Protocol], datasets: Sequence[Dataset],
               overrides: Optional[Dict[str, Any]] = None) -> List[Run]:
    """M×N 笛卡尔积: 每个 protocol × 每个 dataset 生成一个 Run。"""
    runs = []
    for p in protocols:
        for d in datasets:
            runs.append(Run(protocol=p, dataset=d,
                            run_id=f"{p.name}__{d.name}",
                            overrides=dict(overrides or {})))
    return runs


__all__ = ["Protocol", "Dataset", "Run", "run_matrix"]
