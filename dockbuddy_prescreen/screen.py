# -*- coding: utf-8 -*-
"""
screen —— 应用层(三层架构的顶层): 一键端到端突变筛选
================================================================================
三层架构:
  应用层(本模块)  screen_mutations(序列) -> 少量湿实验候选
  业务层         orchestrator.run_pipeline(按配置编排组件)
  组件层         components(各模型/组件统一编程接口)

设计目标: 输入一条野生型序列, 自动完成「饱和突变 → 打分 → 硬约束 → 多样性采样 →
精筛 → 导出」, 最终只留下少量高潜力突变供湿实验。中间每一步的模型/阈值都由
配置驱动, 可替换。

用法:
    from .screen import screen_mutations
    result = screen_mutations(
        "MSEQN...",                                   # 野生型序列
        enzyme_id="my_enzyme",
        sites=[126, 174, 204],                        # 只突催化位点(缺省全序列饱和)
        scoring={"adapter": "esm1v", "transport": "remote",
                 "instance_id": "ins-xxx", ...},      # 打分模型(可按配置换)
        constraints={"foldx_ddg": {"max": 2.0}},      # 硬约束
        n_select=30, n_final=10,                      # 一层采样 30, 二层精筛留 10
    )
    print(result["candidates"])   # 湿实验候选
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .mutlib import saturation_library
from .orchestrator import run_pipeline


def screen_mutations(sequence: str, *,
                     enzyme_id: str = "",
                     sites: Optional[Sequence[int]] = None,
                     scoring: Optional[Dict[str, Any]] = None,
                     constraints: Optional[Dict[str, Any]] = None,
                     n_select: int = 30,
                     n_final: int = 10,
                     refine_objectives: Optional[Sequence[Dict[str, Any]]] = None,
                     exclude: Sequence[str] = ("C",),
                     out_dir: str = "",
                     dry_run: bool = False) -> Dict[str, Any]:
    """一键突变筛选: 序列 → 饱和突变 → 打分 → 硬约束 → 采样 → 精筛 → 湿实验候选。

    参数:
      sequence: 野生型氨基酸序列(1-letter)。
      enzyme_id: 体系名(产物目录/标识)。
      sites: 限定位点(1-based 残基号列表)。缺省 None = 全序列饱和突变。
      scoring: 打分配置(adapter + transport + 各模型参数)。缺省本地 esm1v。
      constraints: libgate 硬约束 {metric: {"min"/"max": 值}}。缺省 None = 不淘汰。
      n_select: 第一层多样性采样个数(从 PASS 集合挑)。
      n_final: 第二层精筛后最终保留的候选数。
      refine_objectives: 第二层软排序目标 [{name, key, direction}]。
      exclude: 排除的氨基酸(缺省排除 Cys, 避免二硫键干扰)。
      out_dir: 产物目录(缺省走 DOCK_OUTPUT 约定)。
      dry_run: 只演练不落盘。

    返回:
      {candidates: [{variant_id, site, from, to, proxy_scores}],  # 最终湿实验候选
       summary: {...}}                                             # run_pipeline 的完整 summary
    """
    seq = "".join(str(sequence).split()).upper()
    if not seq:
        raise ValueError("sequence 不能为空")
    if sites is None:
        sites = list(range(1, len(seq) + 1))
    ms = saturation_library(seq, [int(s) for s in sites],
                            enzyme_id=enzyme_id, exclude=exclude)

    manifest: Dict[str, Any] = {
        "enzyme_id": enzyme_id,
        "mutationset": ms.to_dict(),
        "scoring": scoring or {"adapter": "esm1v", "transport": "local"},
        "constraints": constraints or {},
        "diversity": {"n_select": n_select, "features": ["site", "aa_type"]},
        "export": {"format": "evolvepro", "label_source": "proxy",
                   "predicts_activity": False},
    }
    # scores_file 是 orchestrator 的**顶层** manifest 字段(与 scoring.adapter=scores_file
    # 配套), 而本 API 的用户会在 scoring 里传 scores_file —— 这里提升到顶层, 否则
    # adapter=scores_file 会读空文件静默不打分(业务口径不一致的隐蔽 bug)。
    if scoring and scoring.get("scores_file"):
        manifest["scores_file"] = scoring["scores_file"]
    if n_final and n_final < n_select:
        manifest["refine"] = {
            "n_final": n_final,
            "objectives": refine_objectives or _default_objectives(scoring),
        }

    summary = run_pipeline(manifest, dry_run=dry_run, out_dir=out_dir)
    candidates = summary.get("refine", {}).get("final") or summary.get("selected", [])
    return {"candidates": candidates, "summary": summary}


def _default_objectives(scoring: Optional[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
    """按打分 adapter 给缺省软排序目标(score 越小越「进化耐受」, 方向按需)。"""
    adapter = (scoring or {}).get("adapter", "esm1v")
    key = {"foldx": "foldx_ddg", "rosetta": "rosetta_ddg"}.get(adapter, f"{adapter}_score")
    return [{"name": "proxy", "key": key, "direction": "min"}]


__all__ = ["screen_mutations"]
