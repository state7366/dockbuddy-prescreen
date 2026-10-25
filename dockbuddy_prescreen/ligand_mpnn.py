# -*- coding: utf-8 -*-
"""prescreen.ligand_mpnn —— LigandMPNN 配体感知反向折叠重设计(2026-10-11 新增)
===================================================================================
LigandMPNN(Dauparas et al. 2023)是 ProteinMPNN 的配体感知版, 重设计时考虑配体
上下文(对「扩口袋对口/底物结合」方向对口)。调用同 ProteinMPNN, 但需配体 PDB +
LigandMPNN 权重(~5MB/个)。

诚实边界: 需 LigandMPNN 权重 + 配体 PDB; 本适配器复用 scoring.mpnn 环境, 配体
上下文通过 ligand_mpnn 特有参数传入(骨架, 部署后需实测对齐)。
"""
from __future__ import annotations

from typing import Any, Dict, List


def redesign_sequences(wt_seq: str, *, receptor: str = "", chain: str = "A",
                       ligand_pdb: str = "", repo_dir: str = "",
                       checkpoint: str = "", fixed: List[int] = (),
                       num_seqs: int = 1, temperature: float = 0.1,
                       seed: int = 0) -> Dict[str, Any]:
    """LigandMPNN 配体感知反向折叠重设计。

    参数: receptor(蛋白骨架 PDB), ligand_pdb(配体 PDB, 必填), checkpoint(LigandMPNN
    权重路径), 其余同 ProteinMPNN。
    """
    if not receptor:
        raise RuntimeError("LigandMPNN 需要 receptor(蛋白骨架 PDB)")
    if not ligand_pdb:
        raise RuntimeError(
            "LigandMPNN 需要 ligand_pdb(配体 PDB; 配体感知重设计的上下文)")
    if not checkpoint:
        raise RuntimeError(
            "LigandMPNN 需要 checkpoint 权重路径(见 github.com/dauparas/LigandMPNN)")

    from dockbuddy_scoring.mpnn import redesign_sequences as _base
    # 配体上下文: 骨架版复用 ProteinMPNN 流程(配体感知的完整接线待实测对齐)
    return _base(wt_seq, receptor=receptor, chain=chain, repo_dir=repo_dir,
                 checkpoint=checkpoint, fixed=list(fixed), num_seqs=num_seqs,
                 temperature=temperature, seed=seed,
                 extra={"ligand_pdb": ligand_pdb})


__all__ = ["redesign_sequences"]
