# -*- coding: utf-8 -*-
"""
libstore.py —— 候选库状态机 + candidate_store.json 文件 IO(预筛 MVP, 2026-10-08)
===================================================================================
定位: 预筛管线的候选状态机。每个候选从 pending 一路推进到 exported / wet_feedback,
中途可能被 rejected(布尔闸)或 discordant(湿实验背离)。状态与推进规则集中在这里,
orchestrator 只负责调度, 不改状态语义。

状态机(单链推进, 末态终局):
    pending → scored → passed_gate → selected → exported → wet_feedback
                  ↘ rejected          (末态)              ↘ discordant (末态)

   - rejected      = 布尔闸未过(硬约束不满足 / 指标缺失)。
   - discordant    = 湿实验回流时, gate 判定与 wet_verdict 相反(代理指标假阳/假阴)。
   - wet_feedback  = 湿实验数据回流且与 gate 判定一致。

关键约定:
  - **唯一键 key = 残基+位点+目标 (+:链)**, 即 f"{from}{site}{to}" + (":{chain}" if chain)。
    variant_id 只是人类标签(不含链), 跨链会重名, 故索引一律用 key。
  - 状态推进为**幂等/宽容**: 越序或重复调用只跳过+记日志, 不 raise(编排要鲁棒)。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

LOG = logging.getLogger("dockpipeline")

SCHEMA_VERSION = "1.0"
STATES = ("pending", "scored", "passed_gate", "selected", "exported",
          "wet_feedback", "rejected", "discordant")


def variant_key(site, from_aa: str, to_aa: str, chain: str = "") -> str:
    return f"{from_aa}{site}{to_aa}" + (f":{chain}" if chain else "")


class CandidateStore:
    """候选库: 有序 dict(key -> record)。record 为纯 dict(JSON 友好)。"""

    def __init__(self, enzyme_id: str = "", template_sequence: str = "",
                 records: Optional[Dict[str, Dict[str, Any]]] = None):
        self.enzyme_id = enzyme_id
        self.template_sequence = template_sequence
        self.records: Dict[str, Dict[str, Any]] = records if records is not None else {}

    # -- 增删 -----------------------------------------------------------
    def add_variant(self, site, from_aa: str, to_aa: str, chain: str = "",
                    source: str = "") -> str:
        key = variant_key(site, from_aa, to_aa, chain)
        if key in self.records:
            return key                              # 幂等
        self.records[key] = {
            "key": key,
            "variant_id": f"{from_aa}{site}{to_aa}",
            "site": int(site), "from": from_aa, "to": to_aa, "chain": chain,
            "source": source,
            "status": "pending",
            "proxy_scores": {},
            "gate": None,
            "wet": None,
            "discrepancy_flag": False,
        }
        return key

    def add_mutation_set(self, ms) -> List[str]:
        """从 libmutlib.MutationSet(或任意可迭代 Variant)批量添加。"""
        keys = []
        for v in ms:
            keys.append(self.add_variant(
                v.site, v.from_aa, v.to_aa, v.chain, getattr(v, "source", "")))
        return keys

    # -- 查询 -----------------------------------------------------------
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.records.get(key)

    def keys(self) -> List[str]:
        return list(self.records.keys())

    def by_status(self, status: str) -> List[Dict[str, Any]]:
        return [r for r in self.records.values() if r["status"] == status]

    def by_variant_id(self, variant_id: str) -> Optional[Dict[str, Any]]:
        for r in self.records.values():
            if r["variant_id"] == variant_id:
                return r
        return None

    def counts(self) -> Dict[str, int]:
        c = {s: 0 for s in STATES}
        for r in self.records.values():
            c[r["status"]] = c.get(r["status"], 0) + 1
        c["total"] = len(self.records)
        return c

    # -- 状态推进 -------------------------------------------------------
    def score(self, key: str, metrics: Dict[str, Any]) -> bool:
        r = self.records.get(key)
        if r is None:
            return False
        r["proxy_scores"] = dict(metrics or {})
        if r["status"] in ("pending", "scored"):
            r["status"] = "scored"
        return True

    def apply_gate(self, key: str, gate_result) -> bool:
        r = self.records.get(key)
        if r is None:
            return False
        r["gate"] = gate_result.to_dict() if hasattr(gate_result, "to_dict") \
            else dict(gate_result)
        r["status"] = "passed_gate" if bool(r["gate"].get("passed")) else "rejected"
        return True

    def select(self, keys: Iterable[str]) -> None:
        s = set(keys)
        for r in self.records.values():
            if r["key"] in s:
                r["status"] = "selected"

    def mark_exported(self, keys: Iterable[str]) -> None:
        s = set(keys)
        for r in self.records.values():
            if r["key"] in s:
                r["status"] = "exported"

    def apply_feedback(self, key: str, wet: Dict[str, Any]) -> bool:
        """湿实验回流: 记 wet 数据, 按 gate 判定 vs wet_verdict 判 discordant。"""
        r = self.records.get(key)
        if r is None:
            return False
        r["wet"] = dict(wet or {})
        gate_passed = bool((r.get("gate") or {}).get("passed", False))
        verdict = str((wet or {}).get("verdict", "")).strip().lower()
        disc = (gate_passed and verdict in ("inactive", "no", "negative")) or \
               ((not gate_passed) and verdict in ("active", "yes", "positive"))
        r["discrepancy_flag"] = bool(disc)
        r["status"] = "discordant" if disc else "wet_feedback"
        return True

    # -- 序列化 ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enzyme_id": self.enzyme_id,
            "template_sequence": self.template_sequence,
            "records": list(self.records.values()),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateStore":
        st = cls(d.get("enzyme_id", ""), d.get("template_sequence", ""))
        for r in d.get("records", []):
            st.records[r["key"]] = r
        return st

    def write_json(self, path: str) -> str:
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    @classmethod
    def read_json(cls, path: str) -> "CandidateStore":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


__all__ = ["SCHEMA_VERSION", "STATES", "variant_key", "CandidateStore"]
