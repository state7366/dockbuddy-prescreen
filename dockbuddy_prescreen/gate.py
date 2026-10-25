# -*- coding: utf-8 -*-
"""
libgate.py —— 布尔硬约束闸(预筛管线 MVP 决策核心, 2026-10-08 新增)
===================================================================================
定位: 预筛管线「布尔硬约束筛选」的唯一实现。它只回答一个问题 —— 给定某候选的
一组代理指标, 该候选是否**通过**全部硬约束。它**不做排序、不打总分、不输出排名**。

设计决策(由用户拍板, 记录备查):
  - **扁平阈值表**(非组合布尔表达式): manifest 用 `{"metric": {"min"/"max"}}` 表达,
    所有约束隐含 AND。暂不支持 OR / "M 中至少 N 个"(那是全量阶段的组合布尔)。
  - **单终端闸**(非多阶段闸): gate 只对「已打分候选集」做一次过滤; 将来要分阶段
    早杀时再给 manifest 加 stage 维度(届时本模块的 evaluate 签名向后兼容)。
  - **阈值全部外置**: 本模块不内置任何阈值; manifest 没写约束 = 全通过(空真)。
  - **rank 隔离铁律**: 本模块**禁止 import 任何 rank/排序模块**(见 test_gate_rank_isolated)。
    排序(rank)只是内部诊断, 绝不参与候选筛选; 若在此引入 rank, 会破坏
    「布尔硬约束 + 多样性采样」的核心哲学。
  - **代理指标不自称活性**: 通过闸 = "满足硬约束、值得进入下一轮", 绝不等于
    "活性更高 / kcat 更高"。

门控语义:
  - 被约束指标缺失(required=True, 默认) → 该约束失败(reason='missing')。
  - 值为 None / bool / 非数值 → 失败(reason='non_numeric')。
  - 值在 [min,max] 内(单边时只查一边) → 通过。
  - constraints 为空 → 空真, 全体通过(此时闸退化为纯枚举)。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("dockpipeline")

SCHEMA_VERSION = "1.0"
ALLOWED_SPEC_KEYS = ("min", "max", "required")


# ----------------------------------------------------------------------------
# 结果对象
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    """单条约束的判定结果。"""
    metric: str
    value: Any
    min: Optional[float]
    max: Optional[float]
    ok: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"metric": self.metric, "value": self.value,
                "min": self.min, "max": self.max,
                "ok": self.ok, "reason": self.reason}


@dataclass(frozen=True)
class GateResult:
    """单个候选过闸结果。passed = 所有约束均为真(空约束为真)。"""
    passed: bool
    checks: List[Check]
    missing: List[str] = field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.ok)

    @property
    def n_fail(self) -> int:
        return len(self.checks) - self.n_pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "missing": list(self.missing),
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "checks": [c.to_dict() for c in self.checks],
        }


# ----------------------------------------------------------------------------
# 约束归一化
# ----------------------------------------------------------------------------
def normalize_constraints(constraints: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """校验并归一化扁平阈值表 `{metric: {min?/max?/required?}}`。

    规则(违反即 ValueError):
      - constraints 必须是 dict;
      - 每个 metric 的 spec 必须是 dict 且只含 min/max/required 字段;
      - 每个 metric 至少要有 min 或 max;
      - required 默认 True(缺失即失败)。
    """
    if constraints is None:
        return {}
    if not isinstance(constraints, dict):
        raise ValueError("constraints 必须是 dict(扁平阈值表), 如 {\"foldx_ddg\": {\"max\": 2.0}}")
    out: Dict[str, Dict[str, Any]] = {}
    for metric, spec in constraints.items():
        if not isinstance(spec, dict):
            raise ValueError(f"约束 {metric!r} 必须是 dict, 如 {{\"max\": 2.0}}")
        unknown = set(spec) - set(ALLOWED_SPEC_KEYS)
        if unknown:
            raise ValueError(f"约束 {metric!r} 含未知字段 {sorted(unknown)}"
                             f"(仅允许 {sorted(ALLOWED_SPEC_KEYS)})")
        if "min" not in spec and "max" not in spec:
            raise ValueError(f"约束 {metric!r} 至少需要 min 或 max")
        out[metric] = {
            "min": spec.get("min"),
            "max": spec.get("max"),
            "required": bool(spec.get("required", True)),
        }
    return out


def load_constraints(path: str) -> Dict[str, Any]:
    """从 JSON 文件读取原始约束表(不做归一化, 交 evaluate 统一归一)。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"约束文件 {path} 顶层必须是对象")
    return data


# ----------------------------------------------------------------------------
# 判定
# ----------------------------------------------------------------------------
def _satisfies(value: Any, lo: Optional[float], hi: Optional[float]):
    """返回 (ok, reason)。非数值(None/bool/str)一律视为不可判定。"""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False, "non_numeric"
    if lo is not None and value < lo:
        return False, "below_min"
    if hi is not None and value > hi:
        return False, "above_max"
    return True, ""


def _evaluate(metrics: Dict[str, Any], cons: Dict[str, Dict[str, Any]]) -> GateResult:
    metrics = metrics or {}
    checks: List[Check] = []
    missing: List[str] = []
    for metric, spec in cons.items():
        if metric not in metrics:
            if not spec["required"]:
                continue                        # 可选约束, 缺失不判定
            missing.append(metric)
            checks.append(Check(metric, None, spec["min"], spec["max"], False, "missing"))
            continue
        val = metrics[metric]
        ok, reason = _satisfies(val, spec["min"], spec["max"])
        checks.append(Check(metric, val, spec["min"], spec["max"], ok, reason))
    return GateResult(passed=all(c.ok for c in checks), checks=checks, missing=missing)


def evaluate(metrics: Dict[str, Any], constraints: Dict[str, Any]) -> GateResult:
    """对单个候选的指标 dict 施加扁平阈值表。constraints 每次归一(体积小, 无碍)。"""
    return _evaluate(metrics, normalize_constraints(constraints))


def evaluate_many(metrics_map: Dict[str, Dict[str, Any]],
                  constraints: Dict[str, Any]) -> Dict[str, GateResult]:
    """对一批候选(键 -> 指标 dict)批量过闸, 约束只归一一次。"""
    cons = normalize_constraints(constraints)
    return {k: _evaluate(m, cons) for k, m in metrics_map.items()}


__all__ = [
    "SCHEMA_VERSION",
    "Check",
    "GateResult",
    "normalize_constraints",
    "load_constraints",
    "evaluate",
    "evaluate_many",
]
