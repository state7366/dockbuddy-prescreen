# -*- coding: utf-8 -*-
"""
libexport.py —— EVOLVEpro 格式适配器(预筛 MVP 输出端, 2026-10-08 新增)
===================================================================================
定位: 把「最终选中」的候选导出成下游实验设计工具(EVOLVEpro)可消费的 TSV, 并**显式
标注标签性质** —— label 是计算代理指标(proxy), **不预测 kcat/Km 活性**。这是管线
边界 4 条里的诚实口径落点: 输出只作代理指标, 结论须湿实验收口。

产物(两件, 同 stem):
  - <out>.tsv          列 `variant<TAB>label`; label 取 label_metric 对应的 proxy 值
                       (无 label_metric 时取第一个可用数值分; 缺分则空)。
  - <out>.meta.json    元数据: {label_source, predicts_activity, label_metric, n,
                       n_labelled, disclaimer}。predicts_activity 恒 false(MVP)。

为什么要有 meta.json: 直接把 proxy 数字塞进 EVOLVEpro 会诱导下游把「更负的 ΔΔG /
更低的对接能」读成「更高活性」。meta 里的 label_source=proxy + predicts_activity=
false 是防误读的强制标记, 也是 export 阶段自否证的一部分。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

LOG = logging.getLogger("dockpipeline")

DISCLAIMER = ("本标签为计算代理指标(proxy), 不预测 kcat/Km 活性; "
              "EVOLVEpro 仅作候选优先级启发, 结论须湿实验收口。")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pick_label(r: Dict[str, Any], label_metric: str) -> Optional[Any]:
    scores = r.get("proxy_scores", {}) or {}
    if label_metric and label_metric in scores and _is_number(scores[label_metric]):
        return scores[label_metric]
    if not label_metric:                       # 兜底: 取第一个数值分
        for v in scores.values():
            if _is_number(v):
                return v
    return None


def export_evolvepro(records: Iterable[Dict[str, Any]],
                     out_path: str,
                     *,
                     label_metric: str = "",
                     label_source: str = "proxy",
                     predicts_activity: bool = False) -> Dict[str, Any]:
    """导出选中候选为 EVOLVEpro TSV + 元数据 JSON。返回产物路径与统计。"""
    recs: List[Dict[str, Any]] = list(records)
    base = os.path.splitext(out_path)[0]
    tsv_path = out_path if out_path.lower().endswith((".tsv", ".csv", ".txt")) \
        else base + ".tsv"
    meta_path = base + ".meta.json"

    rows: List[Dict[str, Any]] = []
    n_labelled = 0
    for r in recs:
        vid = r.get("variant_id") or r.get("key") or ""
        label = _pick_label(r, label_metric)
        if label is not None:
            n_labelled += 1
        rows.append({"variant": vid, "label": label})

    os.makedirs(os.path.dirname(os.path.abspath(tsv_path)) or ".", exist_ok=True)
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("variant\tlabel\n")
        for row in rows:
            lab = "" if row["label"] is None else row["label"]
            f.write(f"{row['variant']}\t{lab}\n")

    meta = {
        "label_source": label_source,
        "predicts_activity": bool(predicts_activity),
        "label_metric": label_metric,
        "n": len(rows),
        "n_labelled": n_labelled,
        "disclaimer": DISCLAIMER,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "out_tsv": tsv_path, "out_meta": meta_path,
        "n": len(rows), "n_labelled": n_labelled,
        "label_metric": label_metric,
        "label_source": label_source,
        "predicts_activity": bool(predicts_activity),
    }


__all__ = ["export_evolvepro", "DISCLAIMER"]
