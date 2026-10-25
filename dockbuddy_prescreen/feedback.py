# -*- coding: utf-8 -*-
"""
libfeedback.py —— 失败日志 + 湿实验回流(预筛 MVP 强制模块, 2026-10-08 新增)
===================================================================================
定位: 预筛管线的两个「闭环」落点:
  1. **failure_log.json** —— 强制模块。每个被布尔闸剔除、或湿实验背离 gate 判定的
     候选都留一条失败记录(含 variant_id / stage / reason / proxy_scores /
     gate_version / timestamp)。这是「实验数据回流后校准阈值」的数据基础 —— 没有
     它, 阈值外置就只是一次性拍脑袋, 无法据假阳/假阴率回调。
  2. **湿实验回流 ingest_feedback** —— 把 wet_verdict(active/inactive)写回候选,
     与 gate 判定对撞:
       gate=passed 但 wet=inactive → discordant(代理假阳)
       gate=rejected 但 wet=active → discordant(代理假阴)
     两者都记 failure_log, 供下一轮阈值校准; 一致则记为 wet_feedback。

设计决策(记录备查):
  - gate_version 字段: 每次阈值调整都 bump, 使 failure_log 可追溯「哪一版闸」造成的
    假阳/假阴 —— 阈值校准的必备元数据。
  - 湿实验口径只存 verdict 字符串(active/inactive/...)+ 可选 kcat_km, 不与代理指标
    在数值尺度上强行对齐(代理 ΔΔG 与湿 kcat/Km 不同纲, 硬比会制造假精度)。
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("dockpipeline")

SCHEMA_VERSION = "1.0"
GATE_VERSION = "mvp-1.0"

_WET_NEGATIVE = ("inactive", "no", "negative")
_WET_POSITIVE = ("active", "yes", "positive")


class FailureLog:
    """失败日志: 纯追加列表, 落盘 failure_log.json。"""

    def __init__(self, entries: Optional[List[Dict[str, Any]]] = None):
        self.entries: List[Dict[str, Any]] = entries if entries is not None else []

    def append(self, *, variant_id: str, stage: str, reason: str,
               gate_version: str = GATE_VERSION,
               proxy_scores: Optional[Dict[str, Any]] = None,
               timestamp: str = "") -> None:
        self.entries.append({
            "variant_id": variant_id,
            "stage": stage,
            "reason": reason,
            "gate_version": gate_version,
            "proxy_scores": proxy_scores or {},
            "timestamp": timestamp or datetime.datetime.now().isoformat(timespec="seconds"),
        })

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "gate_version": GATE_VERSION,
                "entries": self.entries}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FailureLog":
        return cls(entries=d.get("entries", []))

    def write_json(self, path: str) -> str:
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    @classmethod
    def read_json(cls, path: str) -> "FailureLog":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def ingest_feedback(store, feedback_entries: List[Dict[str, Any]],
                    failure_log: Optional[FailureLog] = None,
                    gate_version: str = GATE_VERSION) -> Dict[str, Any]:
    """湿实验回流: 逐条写回 store, 背离者记 failure_log。返回汇总。

    feedback_entries 每条: {variant_id 或 key, verdict: active/inactive, kcat_km?: ...}。
    """
    discordant: List[str] = []
    n_applied = 0
    for fe in feedback_entries:
        key = str(fe.get("key") or fe.get("variant_id") or "")
        rec = store.get(key) if store.get(key) is not None else store.by_variant_id(key)
        if rec is None:
            LOG.warning("feedback 命中未知候选, 跳过: %s", key)
            continue
        wet = dict(fe.get("wet") or {
            "verdict": fe.get("verdict", ""),
            "kcat_km": fe.get("kcat_km"),
        })
        store.apply_feedback(rec["key"], wet)
        n_applied += 1
        if rec["discrepancy_flag"]:
            discordant.append(rec["variant_id"])
            if failure_log is not None:
                failure_log.append(
                    variant_id=rec["variant_id"], stage="wet_feedback",
                    reason=f"gate 判定与湿实验背离(verdict={wet.get('verdict', '')})",
                    gate_version=gate_version,
                    proxy_scores=rec.get("proxy_scores"))
    return {"n_feedback": len(feedback_entries), "n_applied": n_applied,
            "n_discordant": len(discordant), "discordant": discordant}


__all__ = ["SCHEMA_VERSION", "GATE_VERSION", "FailureLog", "ingest_feedback"]
