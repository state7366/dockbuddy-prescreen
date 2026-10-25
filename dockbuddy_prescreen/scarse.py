# -*- coding: utf-8 -*-
"""
prescreen/scarse.py —— SCARSE 式「CV R² 早诊」主动学习前置质检(2026-10-23 新增)
====================================================================================
定位: 落地 ima「论文主动发现」新增论文 SCARSE(bioRxiv 2026.06.25.734678) 的核心
方法学增量 —— 用「交叉验证 R²(CV R²)」作为「该特征/模型是否值得投入湿实验主动
学习」的早诊指标。SCARSE 原文的回归器(ESM-2 embedding + 高斯过程)与 DockBuddy
既有 `active_learning.py`(GP+UCB/EI) 同源, 故本模块**不重复造回归器**, 只补
「CV R² 诊断」这一增量, 复用 active_learning 的 GP 后验。

原文关键结论(本模块复现的命题):
  1. CV R² 与最终主动学习性能强相关(原文 Pearson r = 0.80~0.84);
  2. 评估少量(约 50 条)已标注样本即可把 CV R² 方差压到可用水平 —— 即可提前、
     低成本判断「该特征/模型是否适合驱动主动学习」。

设计决策(记录备查):
  - 纯 numpy 实现 K 折 CV R², 零额外依赖(不引入 sklearn), 与 active_learning.py
    纪律一致(mock 单测无 GPU 可绿)。
  - 复用 `active_learning.gp_posterior` 做折内预测, 保持「GP 内核」单一事实源。
  - R² 口径: 折内 test 集 R² = 1 - SS_res/SS_tot, SS_tot 以 test 折真实 y 均值
    为基准(与 sklearn r2_score 一致), 跨折取 mean±std 返回。
  - 小样本护栏: 折数 k 自适应 min(k, n), 每折至少留 2 个测试样本, 否则无法定义
    折内方差。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .active_learning import gp_posterior

__all__ = ["gp_cv_rsq", "assess_al_readiness", "R2FOLD_MIN_TEST"]


# 每折测试集最少样本数(少于该数时该折 R² 不可靠, 记为 NaN 并跳过)。
R2FOLD_MIN_TEST = 2


def _fold_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """单折 R² = 1 - SS_res/SS_tot, SS_tot 以 y_true 均值为基准。"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def gp_cv_rsq(X: np.ndarray, y: np.ndarray, k: int = 5, *,
              lengthscale: float = 1.0, noise: float = 1e-3,
              seed: int = 0) -> Tuple[float, float, List[float]]:
    """K 折交叉验证的 GP 回归 R²。

    返回 (r2_mean, r2_std, fold_r2s)。fold_r2s 中不可靠的折记为 NaN。
    - k 自适应 min(k, n // (R2FOLD_MIN_TEST + 1)), 保证每折至少 2 个测试样本;
      n 过小(< 3)时无法做留出 CV, 返回 (NaN, NaN, [])。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    if n < 3:
        return float("nan"), float("nan"), []

    k = max(1, min(k, n // (R2FOLD_MIN_TEST + 1)))
    k = min(k, n)  # 折数不超过样本数(退化为 leave-one-out 上界)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    fold_r2s: List[float] = []
    for fold in folds:
        test_idx = np.asarray(fold, dtype=int)
        if test_idx.size < R2FOLD_MIN_TEST:
            fold_r2s.append(float("nan"))
            continue
        train_idx = np.setdiff1d(perm, test_idx)
        if train_idx.size == 0:
            fold_r2s.append(float("nan"))
            continue
        mean, _ = gp_posterior(X[train_idx], y[train_idx], X[test_idx],
                               lengthscale=lengthscale, noise=noise)
        fold_r2s.append(_fold_r2(y[test_idx], mean))

    valid = [r for r in fold_r2s if not np.isnan(r)]
    if not valid:
        return float("nan"), float("nan"), fold_r2s
    r2_mean = float(np.mean(valid))
    r2_std = float(np.std(valid)) if len(valid) > 1 else 0.0
    return r2_mean, r2_std, fold_r2s


def assess_al_readiness(X: np.ndarray, y: np.ndarray, k: int = 5, *,
                        lengthscale: float = 1.0, noise: float = 1e-3,
                        seed: int = 0, threshold: float = 0.3) -> dict:
    """SCARSE 式早诊入口: 给 (特征, 已测活性), 输出是否值得投入主动学习的判据。

    返回 dict:
      - cv_rsq_mean / cv_rsq_std: CV R² 均值/标准差;
      - n_samples: 已标注样本数;
      - verdict: 'ready' | 'marginal' | 'poor'(CV R² 阈值判定, 阈值可调);
      - note: 文本结论(供日志/报告)。
    """
    r2_mean, r2_std, folds = gp_cv_rsq(X, y, k=k, lengthscale=lengthscale,
                                        noise=noise, seed=seed)
    n = int(np.atleast_2d(np.asarray(X)).shape[0])
    if np.isnan(r2_mean):
        verdict, note = "poor", "样本数不足(<3), 无法计算可靠 CV R², 建议先补充标注。"
    elif r2_mean >= threshold:
        verdict = "ready"
        note = (f"CV R²={r2_mean:.3f}±{r2_std:.3f} ≥ {threshold}, "
                f"该特征/模型值得投入主动学习。")
    elif r2_mean >= threshold / 2:
        verdict = "marginal"
        note = (f"CV R²={r2_mean:.3f}±{r2_std:.3f} 介于阈值[{threshold/2:.2f},"
                f"{threshold:.2f}), 特征信号中等, 建议谨慎评估或换特征。")
    else:
        verdict = "poor"
        note = (f"CV R²={r2_mean:.3f}±{r2_std:.3f} < {threshold/2:.2f}, "
                f"特征几乎无适应度信号, 不建议投入主动学习。")
    return {
        "cv_rsq_mean": r2_mean,
        "cv_rsq_std": r2_std,
        "fold_rsq": folds,
        "n_samples": n,
        "verdict": verdict,
        "note": note,
    }
