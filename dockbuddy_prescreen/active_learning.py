# -*- coding: utf-8 -*-
"""
prescreen/active_learning.py —— 主动学习定向进化编排层(EVOLVEpro 式, 2026-10-12 新增)
===================================================================================
定位: 接入 EVOLVEpro(Jiang et al. 2025, Science 387:adr6006)的 few-shot 主动学习
闭环, 作为管线「主动学习定向进化」环节。此前因「纯计算管线、无湿实验反馈环」判
不接入; 现湿实验条件已具备, 本模块把「计算打分 → 湿实验验证 → 回喂训练 → 再选点」
闭环的计算侧落地, 湿实验活性数据经 `feedback.ingest_feedback` 回流后作为回归标签。

核心算法(EVOLVEpro 同款范式, 轻量实现):
  1. 特征 = 蛋白语言模型 embedding(如 ESM-2 顶层表示)对每个候选的编码。
  2. 回归 = 高斯过程(GP, RBF 核)在已测 (特征, 湿实验活性) 对上拟合, 给出每个未测
     候选的预测均值与不确定度 —— 相比普通岭回归, GP 提供选点所需的方差。
  3. 采集 = UCB(mean + κ·std)或 EI(expected improvement), 选信息量最高的 ~16 个
     候选进入下一轮湿实验(原文 few-shot 口径每轮约 16 个突变)。

与 feedback.py 的关系: feedback 负责「湿实验判定写回 + 假阳/假阴对撞」, 本模块负责
「给定已测数据后, 提议下一轮该测哪些候选」。二者串联即完整主动学习闭环。

设计决策(记录备查):
  - 纯 numpy/scipy 实现 GP, 不依赖 sklearn(本机未装), 零额外依赖, mock 单测可绿。
  - 特征 embedding 由调用方供给(X), 本模块不内置 PLM 加载 —— 保持与 scoring 层解耦,
    便于把 DockBuddy 现有多模型打分/任意 PLM embedding 作为特征注入。
  - 数值稳定: 核矩阵 + 噪声项正则化 + 对称化, 避免半正定破缺。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["gp_posterior", "ucb", "expected_improvement", "propose_next_round",
           "run_active_loop"]


def _rbf_kernel(A: np.ndarray, B: np.ndarray, lengthscale: float) -> np.ndarray:
    """RBF 核 K(A,B)[i,j] = exp(-0.5 ||A_i - B_j||^2 / ls^2)。"""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    sq = (A[:, None, :] - B[None, :, :]) ** 2
    d2 = sq.sum(axis=-1)
    return np.exp(-0.5 * d2 / (lengthscale * lengthscale))


def gp_posterior(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                 lengthscale: float = 1.0, noise: float = 1e-3,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """GP 后验: 返回 (mean, std) 对 X_test 的预测分布。y 按零均值先验中心化。"""
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
    y = np.asarray(y_train, dtype=float).ravel()
    y = y - y.mean()  # 零均值 GP 先验

    n = X_train.shape[0]
    K = _rbf_kernel(X_train, X_train, lengthscale) + noise * np.eye(n)
    K = 0.5 * (K + K.T)  # 对称化保半正定
    K_inv = np.linalg.inv(K)

    K_s = _rbf_kernel(X_test, X_train, lengthscale)
    mean = K_s @ K_inv @ y + (np.asarray(y_train, dtype=float).ravel().mean())

    K_ss = _rbf_kernel(X_test, X_test, lengthscale)
    cov = K_ss - K_s @ K_inv @ K_s.T
    cov = 0.5 * (cov + cov.T)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return mean, std


def ucb(mean: np.ndarray, std: np.ndarray, kappa: float = 1.96) -> np.ndarray:
    """上置信界(UCB)采集: 均值 + κ·不确定度。"""
    return np.asarray(mean) + kappa * np.asarray(std)


def expected_improvement(mean: np.ndarray, std: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """期望提升(EI)采集。"""
    from scipy.stats import norm
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    z = (mean - y_best - xi) / (std + 1e-12)
    return (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)


def propose_next_round(X_labeled: np.ndarray, y_labeled: np.ndarray,
                       X_pool: np.ndarray, n_select: int = 16, *,
                       acquisition: str = "ucb", kappa: float = 1.96,
                       lengthscale: float = 1.0, noise: float = 1e-3,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提议下一轮要测的候选: 返回 (选中索引, 预测均值, 预测不确定度)。

    参数:
      X_labeled: 已测候选的特征矩阵 (n_labeled, d)。
      y_labeled: 已测候选的湿实验活性 (n_labeled,)。
      X_pool: 未测候选特征矩阵 (n_pool, d)。
      n_select: 每轮选取个数(原文 few-shot 口径约 16)。
      acquisition: 'ucb' 或 'ei'。
    """
    mean, std = gp_posterior(X_labeled, y_labeled, X_pool,
                             lengthscale=lengthscale, noise=noise)
    if acquisition == "ei":
        acq = expected_improvement(mean, std, float(np.asarray(y_labeled).max()))
    else:
        acq = ucb(mean, std, kappa=kappa)
    idx = np.argsort(-acq)[: min(n_select, len(X_pool))]
    return idx, mean, std


def run_active_loop(X_pool: np.ndarray, oracle: "callable", n_rounds: int,
                    n_per_round: int = 16, *, seed: int = 0,
                    acquisition: str = "ucb", lengthscale: float = 1.0,
                    noise: float = 1e-3, kappa: float = 1.96,
                    ) -> List[dict]:
    """完整主动学习闭环(供演示/验证): oracle 模拟湿实验读数, 返回每轮记录。

    oracle(pool_indices) -> np.ndarray: 给定候选全局索引, 返回其真实活性(湿实验)。
    首轮无已测数据时, 随机种子选 n_per_round 个启动(与 EVOLVEpro 的 few-shot 启动
    一致)。每轮记录 {round, selected, best_so_far, regret}。
    """
    rng = np.random.default_rng(seed)
    X_pool = np.atleast_2d(np.asarray(X_pool, dtype=float))
    n_pool = X_pool.shape[0]
    labeled: List[int] = []
    records: List[dict] = []
    prev_best = -np.inf

    for r in range(n_rounds):
        if not labeled:
            sel = rng.choice(n_pool, size=min(n_per_round, n_pool), replace=False).tolist()
        else:
            X_lab = X_pool[np.array(labeled)]
            y_lab = np.array([oracle([i])[0] for i in labeled])
            remaining = [i for i in range(n_pool) if i not in labeled]
            idx, _, _ = propose_next_round(
                X_lab, y_lab, X_pool[np.array(remaining)],
                n_select=n_per_round, acquisition=acquisition,
                kappa=kappa, lengthscale=lengthscale, noise=noise)
            sel = [remaining[i] for i in idx]
        labeled.extend(sel)
        y_now = np.array([oracle([i])[0] for i in labeled])
        best_so_far = float(y_now.max())
        records.append({
            "round": r + 1, "n_labeled": len(labeled), "selected": sel,
            "best_so_far": round(best_so_far, 6),
            "improvement": round(best_so_far - max(prev_best, best_so_far), 6)
            if prev_best != -np.inf else 0.0,
        })
        prev_best = max(prev_best, best_so_far)
    return records
