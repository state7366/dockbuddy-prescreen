# -*- coding: utf-8 -*-
"""
orchestrator.py —— 预筛管线最小编排层(`db pipeline run <manifest>` 的 core)
===================================================================================
定位: 把五个 MVP 模块串成一条闭环, 只做「调度 + 状态 + 落盘」, 不含任何打分/几何/
排序口径(那些分别住在 libmutlib/libgate/libdiverse/libstore/libexport/libfeedback)。
打分/折叠的**真实模型适配器**以「可插拔 adapter」挂在这里(见下), 与业务口径解耦。

流水线(单终端闸):
  1. 解析 manifest → 加载 mutationset(路径或 inline)
  2. 建 CandidateStore(resume 时复用旧 store, 不重复添加候选)
  3. 打分(scoring, 可插拔):
       adapter=scores_file  读 manifest.scores_file(预计算代理指标, 缺省/向后兼容)
       adapter=esm1v        调 scoring.esm.esm1v_scores(ESM-1v zero-shot, 需 GPU 镜像)
       adapter=foldx        调 scoring.foldx.score_variants(FoldX ΔΔG, 需 receptor_pdb)
       adapter=evmutation/saprot/venusrem/promep/rosetta/geostab
                            调 scoring.<name>.score_variants(各自依赖, 见 docs)
  4. 布尔闸(libgate): 逐候选 evaluate, 未过 → rejected + 记 failure_log
  5. 多样性采样(libdiverse): 从 passed_gate 挑 n_select
  6. 折叠回测(folding, 可插拔, 可选):
       adapter=esmfold      调 libfold_api.esmfold_pdb 折叠选中候选, 回写 esmfold_plddt
                            (结构回测: 确认候选可折叠, pLDDT 作额外代理指标)
  7. 导出(libexport): selected → evolvepro.tsv(+meta.json, label_source=proxy)
  8. 落盘 candidate_store.json + failure_log.json(dry_run 跳过)

断点续跑: 非 --fresh 且已有 candidate_store.json → 读旧 store, 已过闸/已选中者不重判。
诚实口径: 输出只作代理指标, 不预测 kcat/Km; 该禁令在 export.meta 里显式落盘。

模型介入点(打分 adapter 见 scoring/ 子包, 折叠见 prescreen/fold_api):
  - ESM-1v : 步骤 3 打分(zero-shot 伪对数似然), 值 = esm1v_score。
  - FoldX  : 步骤 3 打分(ΔΔG), 值 = foldx_ddg(需 scoring.receptor_pdb)。
  - Rosetta: 步骤 3 打分(精细 ΔΔG), 值 = rosetta_ddg(需 PyRosetta + scoring.pdb_path)。
  - GeoStab: 步骤 3 打分(GeoDDG/GeoDTm), 值 = geostab_ddg(需 scoring.geostab_dir)。
  - EVmutation/SaProt/VenusREM/ProMEP: 步骤 3 打分, 值 = <name>_score(各自依赖)。
  - ESMFold: 步骤 6 折叠回测(api 公共接口 / local GPU), 值 = esmfold_plddt。
  - ColabFold: 已注册 spec, 折叠实现未接入(可选; ESMFold 已覆盖折叠回测需求)。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Sequence

from . import export as libexport
from . import feedback as libfeedback
from . import gate as libgate
from . import diverse as libdiverse
from . import mutlib as libmutlib
from . import store as libstore

LOG = logging.getLogger("dockpipeline")

DEFAULT_OUT_DIR = "output/pipeline"

_ESM_ENV_HINT = (
    "ESM-1v 打分需要 torch + fair-esm + esm1v 权重; 本机未安装。"
    "请在腾讯云 GPU 镜像(img-50324vse, 见 deploy/)或已装 fair-esm 的环境运行, "
    "或改用 scoring.adapter=scores_file 喂入预计算分数。"
)

# 远程 GPU 打分各模型 → 结果 metric 名(与本地 libscore 一致)
_REMOTE_METRIC = {
    "esm1v": "esm1v_score",
    "saprot": "saprot_score",
    "venusrem": "venusrem_score",
    "promep": "promep_score",
    "evmutation": "evmutation_score",
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是对象")
    return data


def _resolve_mutationset(mspec) -> "libmutlib.MutationSet":
    """manifest.mutationset: 路径(str) 或 inline dict。"""
    if isinstance(mspec, str):
        return libmutlib.MutationSet.read_json(mspec)
    if isinstance(mspec, dict):
        return libmutlib.MutationSet.from_dict(mspec)
    raise ValueError("manifest.mutationset 须为路径字符串或 inline 对象 "
                     "{template_sequence, variants}")


def _score_from_file(store: "libstore.CandidateStore", scores: Dict[str, Any]) -> int:
    """把 scores_file 的 {variant_id: {metric: value}} 灌进 store。返回命中数。"""
    n = 0
    for key, metrics in scores.items():
        rec = store.get(key) if store.get(key) is not None else store.by_variant_id(key)
        if rec is None:
            LOG.warning("scores_file 含未知候选, 跳过: %s", key)
            continue
        store.score(rec["key"], metrics)
        n += 1
    return n


# ---------------------------------------------------------------------------
# 打分适配器(可插拔)
# ---------------------------------------------------------------------------
def _esm_available() -> bool:
    """延迟探测 torch / fair-esm 是否可导入(真打分需二者 + esm1v 权重)。"""
    try:
        import torch  # noqa: F401
        import esm    # noqa: F401
        return True
    except ImportError:
        return False


def _score_esm1v(store: "libstore.CandidateStore", ms: "libmutlib.MutationSet",
                 cfg: Dict[str, Any]) -> Dict[str, Any]:
    """ESM-1v zero-shot 打分: 对突变库每个单点突变算伪对数似然比, 灌入 proxy_scores。

    打分口径见 libscore_esm; 结果 metric 名为 esm1v_score(正值=比 WT 更被接受)。
    """
    if not _esm_available():
        raise RuntimeError(_ESM_ENV_HINT)
    import dockbuddy_scoring.esm as libscore_esm

    wt_seq = ms.template_sequence
    variants = list(ms.variants)
    kwargs: Dict[str, Any] = {
        "models_dir": str(cfg.get("models_dir", "") or ""),
        "device": str(cfg.get("device", "cpu") or "cpu"),
    }
    seeds = cfg.get("seeds")
    if seeds is not None:
        kwargs["seeds"] = tuple(int(s) for s in seeds)

    scores = libscore_esm.esm1v_scores(wt_seq, variants, **kwargs)
    n = 0
    for vid, score in scores.items():
        rec = store.by_variant_id(vid)
        if rec is None:
            LOG.warning("esm1v 打分命中未知候选, 跳过: %s", vid)
            continue
        store.score(rec["key"], {"esm1v_score": float(score)})
        n += 1
    return {"adapter": "esm1v", "metric": "esm1v_score", "n_scored": n}


def _score_foldx(store: "libstore.CandidateStore", ms: "libmutlib.MutationSet",
                 cfg: Dict[str, Any]) -> Dict[str, Any]:
    """FoldX 5.1 BuildModel ΔΔG 打分(本地二进制, 真实可跑)。

    需要 manifest.scoring.receptor_pdb(WT 结构 PDB 绝对路径); 每个突变独立跑
    RepairPDB+BuildModel(慢, 适合精选后精算, 大规模预筛用 esm1v/scores_file)。
    """
    receptor_pdb = str(cfg.get("receptor_pdb", "") or "").strip()
    if not receptor_pdb:
        raise ValueError(
            "scoring.adapter=foldx 需要 scoring.receptor_pdb(WT 结构 PDB 绝对路径); "
            "无结构时可先用 folding.adapter=esmfold 折叠 WT 序列落盘再喂入。")
    import dockbuddy_scoring.foldx as libscore_foldx

    scores = libscore_foldx.score_variants(
        receptor_pdb, list(ms.variants),
        bin_path=str(cfg.get("bin", "") or ""),
        chain=str(cfg.get("chain", "A") or "A"),
        timeout=int(cfg.get("timeout", 600) or 600),
        workdir=str(cfg.get("workdir", "") or ""))
    n = 0
    for vid, metrics in scores.items():
        rec = store.by_variant_id(vid)
        if rec is None:
            LOG.warning("foldx 打分命中未知候选, 跳过: %s", vid)
            continue
        store.score(rec["key"], metrics)
        n += 1
    return {"adapter": "foldx", "metric": "foldx_ddg", "n_scored": n}


def _score_libscore(store: "libstore.CandidateStore", ms: "libmutlib.MutationSet",
                    cfg: Dict[str, Any], adapter: str, metric_name: str) -> Dict[str, Any]:
    """通用延迟-import 打分适配器(evmutation/saprot/venusrem/promep/rosetta/geostab 共用)。

    约定: scoring.<adapter>.py 暴露 `score_variants(wt_seq, variants, **cfg)`
    返回 {variant_id: float}(或 {variant_id: {metric: value}})。本机缺依赖时该模块
    应抛 RuntimeError(带部署指引), 不透传裸 ImportError。
    """
    import importlib

    mod = importlib.import_module(f"dockbuddy_scoring.{adapter}")
    kwargs = {k: v for k, v in cfg.items() if k != "adapter"}
    scores = mod.score_variants(ms.template_sequence, list(ms.variants), **kwargs)
    n = 0
    for vid, val in scores.items():
        rec = store.by_variant_id(vid)
        if rec is None:
            LOG.warning("%s 打分命中未知候选, 跳过: %s", adapter, vid)
            continue
        if isinstance(val, dict):
            store.score(rec["key"], dict(val))
        else:
            store.score(rec["key"], {metric_name: float(val)})
        n += 1
    return {"adapter": adapter, "metric": metric_name, "n_scored": n}


def _score_remote(store: "libstore.CandidateStore", ms: "libmutlib.MutationSet",
                  adapter: str, remote_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """远程 GPU 打分: 通过腾讯云 TAT 在 GPU 实例上真实调用模型, 拿回分数灌入 store。

    adapter ∈ {esm1v, saprot, venusrem, promep, evmutation}; 本机无 GPU/依赖时走这条
    通道(见 libscore_remote)。remote_cfg 见 manifest.scoring.remote。
    """
    import dockbuddy_scoring.remote as libscore_remote

    metric = _REMOTE_METRIC.get(adapter)
    if metric is None:
        raise ValueError(f"adapter={adapter!r} 不支持远程打分 (支持 {sorted(_REMOTE_METRIC)})")
    scores = libscore_remote.score_variants(adapter, ms.template_sequence,
                                            list(ms.variants), remote_cfg)
    n = 0
    for vid, val in scores.items():
        rec = store.by_variant_id(vid)
        if rec is None:
            LOG.warning("远程打分命中未知候选, 跳过: %s", vid)
            continue
        store.score(rec["key"], {metric: float(val)})
        n += 1
    return {"adapter": adapter, "metric": metric, "n_scored": n, "remote": True}


def _score_adapter(store: "libstore.CandidateStore", ms: "libmutlib.MutationSet",
                   manifest: Dict[str, Any]) -> Dict[str, Any]:
    """按 manifest.scoring.adapter 分发打分。缺省 scores_file(向后兼容)。"""
    scoring = manifest.get("scoring", {}) or {}
    adapter = str(scoring.get("adapter", "scores_file") or "scores_file")
    remote_cfg = scoring.get("remote") or None
    if remote_cfg:                     # 远程 GPU 打分通道(真实触发部署调用)
        if adapter in _REMOTE_METRIC:
            return _score_remote(store, ms, adapter, remote_cfg)
        raise ValueError(
            f"scoring.remote 仅支持 {sorted(_REMOTE_METRIC)} 打分, 收到 adapter={adapter!r}")
    if adapter == "scores_file":
        scores_file = (manifest.get("scores_file") or "").strip()
        scores = _load_json(scores_file) if scores_file else {}
        n = _score_from_file(store, scores)
        return {"adapter": "scores_file", "n_scored": n}
    if adapter == "esm1v":
        return _score_esm1v(store, ms, scoring)
    if adapter == "foldx":
        return _score_foldx(store, ms, scoring)
    if adapter == "evmutation":
        return _score_libscore(store, ms, scoring, "evmutation", "evmutation_score")
    if adapter == "saprot":
        return _score_libscore(store, ms, scoring, "saprot", "saprot_score")
    if adapter == "venusrem":
        return _score_libscore(store, ms, scoring, "venusrem", "venusrem_score")
    if adapter == "promep":
        return _score_libscore(store, ms, scoring, "promep", "promep_score")
    if adapter == "rosetta":
        return _score_libscore(store, ms, scoring, "rosetta", "rosetta_ddg")
    if adapter == "geostab":
        return _score_libscore(store, ms, scoring, "geostab", "geostab_ddg")
    raise ValueError(
        f"未知 scoring.adapter: {adapter!r} "
        f"(支持 scores_file/esm1v/foldx/evmutation/saprot/venusrem/promep/rosetta/geostab)")


# ---------------------------------------------------------------------------
# 折叠适配器(可插拔)
# ---------------------------------------------------------------------------
def _mutated_seq(template: str, site: int, to_aa: str) -> str:
    """在模板序列上做单点替换, 返回突变体序列(1-letter)。site 为 1-based。"""
    seq = list(template)
    if site < 1 or site > len(seq):
        raise ValueError(f"位点 {site} 超出模板长度 {len(seq)}")
    seq[site - 1] = to_aa
    return "".join(seq)


def _plddt_from_pdb(pdb_text: str) -> Optional[float]:
    """从 PDB 文本提取全局 pLDDT(ESMFold/AF 把 pLDDT 写在 ATOM 的 B 因子列)。

    取 CA 原子的 B 因子均值(无 CA 时退化为全原子), 无 ATOM 则返回 None。
    """
    def _bfactor(line: str) -> Optional[float]:
        if len(line) < 66:
            return None
        try:
            return float(line[60:66])
        except ValueError:
            return None

    vals = [b for ln in pdb_text.splitlines()
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA"
            for b in [_bfactor(ln)] if b is not None]
    if not vals:
        vals = [b for ln in pdb_text.splitlines() if ln.startswith("ATOM")
                for b in [_bfactor(ln)] if b is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _fold_esmfold(store: "libstore.CandidateStore", ms: "libmutlib.MutationSet",
                  cfg: Dict[str, Any], out_dir: str,
                  fold_keys: Sequence[str],
                  flog: Optional["libfeedback.FailureLog"] = None) -> Dict[str, Any]:
    """ESMFold 折叠回测: 对 fold_keys 的候选折叠突变体序列, 回写 esmfold_plddt + PDB。

    transport: "api"(缺省, 公共 REST API) / "local"(GPU 服务器本地推理)。
    """
    import dockbuddy_prescreen.fold_api as libfold_api

    transport = str(cfg.get("transport", "api") or "api")
    if transport not in ("api", "local"):
        raise ValueError(f"未知 folding.transport: {transport!r} (支持 api/local)")
    proxy = str(cfg.get("proxy", "") or "")
    timeout = int(cfg.get("timeout", 300) or 300)
    device = str(cfg.get("device", "auto") or "auto")
    chunk_size = int(cfg.get("chunk_size", 0) or 0)
    fold_dir = os.path.join(out_dir, str(cfg.get("out_dir", "folds") or "folds"))
    folded: list = []
    for key in fold_keys:
        rec = store.get(key)
        if rec is None:
            continue
        seq = _mutated_seq(ms.template_sequence, int(rec["site"]), str(rec["to"]))
        try:
            if transport == "local":
                pdb = libfold_api.esmfold_pdb_local(seq, device=device,
                                                    chunk_size=chunk_size)
            else:
                pdb = libfold_api.esmfold_pdb(seq, proxy=proxy, timeout=timeout)
        except Exception as e:  # noqa: BLE001 —— 折叠失败不中断整条管线
            LOG.warning("esmfold 折叠失败 %s: %s", rec["variant_id"], e)
            if flog is not None:
                flog.append(variant_id=rec["variant_id"], stage="fold",
                            reason=f"esmfold 折叠失败: {e}")
            continue
        plddt = _plddt_from_pdb(pdb)
        metrics: Dict[str, Any] = {"esmfold_plddt": plddt} if plddt is not None else {}
        if fold_dir:
            os.makedirs(fold_dir, exist_ok=True)
            pdb_path = os.path.join(fold_dir, f"{rec['variant_id']}.pdb")
            with open(pdb_path, "w", encoding="utf-8") as f:
                f.write(pdb)
            metrics["esmfold_pdb"] = pdb_path
        if metrics:
            cur = dict(store.get(key).get("proxy_scores", {}) or {})
            cur.update(metrics)
            store.get(key)["proxy_scores"] = cur
        folded.append(rec["variant_id"])
    return {"adapter": "esmfold", "transport": transport, "metric": "esmfold_plddt",
            "n_folded": len(folded), "fold_dir": fold_dir}


def _refine_stage(store: "libstore.CandidateStore",
                  selected: Sequence[Dict[str, Any]],
                  cfg: Dict[str, Any], features: Sequence[str],
                  n_final: int) -> Dict[str, Any]:
    """第二层精筛: 软指标多目标排序(libdock.ranking) + 二次多样性采样(libdiverse)。

    全量接入(非部分): 在 stage1 硬闸+一次采样选出的子集上, 用软指标(PLM/FoldX/
    AF3-MAS/Rosetta/MM-GBSA/GNINA 等已写入 proxy_scores 的值)做多目标排序, 再
    二次 select_diverse 挑最终湿实验候选。空 cfg(无 refine 段)→ 直接返回原 selected。
    """
    if not cfg or not cfg.get("objectives"):
        return {"enabled": False, "n_final": len(selected),
                "final": list(selected), "ranking": None}
    from dockbuddy_evolution.ranking import ranking_layer

    objectives = cfg.get("objectives") or []
    weights = cfg.get("weights") or None
    items = []
    for r in selected:
        item = {"id": r.get("variant_id") or r.get("key"),
                "key": r["key"], "site": r.get("site"), "to": r.get("to"),
                "from": r.get("from"), "chain": r.get("chain"),
                "metrics": r.get("proxy_scores", {})}
        for k, v in r.get("proxy_scores", {}).items():
            item[k] = v
        items.append(item)
    rank_res = ranking_layer(items, objectives=objectives, weights=weights,
                             method=str(cfg.get("method", "both") or "both"))
    if not rank_res.get("ok"):
        return {"enabled": True, "n_final": len(selected),
                "final": list(selected), "ranking": rank_res}
    ordered = [e["item"] for e in rank_res.get("ranked", [])]
    n_final = int(cfg.get("n_final", n_final) or n_final)
    div = libdiverse.select_diverse(ordered, min(n_final, len(ordered)),
                                    features=features)
    final = div["selected"]
    return {"enabled": True, "n_final": len(final), "final": final,
            "ranking": rank_res, "diversity": div["coverage"]}


def _fold_adapter(store: "libstore.CandidateStore", ms: "libmutlib.MutationSet",
                  cfg: Dict[str, Any], out_dir: str,
                  selected: Sequence[Dict[str, Any]],
                  flog: Optional["libfeedback.FailureLog"] = None) -> Dict[str, Any]:
    """按 manifest.folding.adapter 分发折叠回测。空配置 → 跳过。"""
    adapter = str(cfg.get("adapter", "") or "")
    if not adapter:
        return {"adapter": "", "n_folded": 0}
    fold_target = str(cfg.get("fold_target", "selected") or "selected")
    if fold_target == "selected":
        fold_keys = [r["key"] for r in selected]
    elif fold_target == "passed":
        fold_keys = [r["key"] for r in store.by_status("passed_gate")]
    elif fold_target == "all":
        fold_keys = [r["key"] for r in store.records.values()]
    else:
        raise ValueError(f"未知 folding.fold_target: {fold_target!r}"
                         f" (支持 selected/passed/all)")
    if adapter == "esmfold":
        return _fold_esmfold(store, ms, cfg, out_dir, fold_keys, flog)
    raise ValueError(f"未知 folding.adapter: {adapter!r} (支持 esmfold)")


def _resolve_out_dir(manifest: Dict[str, Any], out_dir: str = "") -> str:
    enzyme_id = str(manifest.get("enzyme_id", "") or "")
    out_dir = (out_dir or manifest.get("out_dir", DEFAULT_OUT_DIR)).strip()
    if enzyme_id and os.path.basename(out_dir) != enzyme_id:
        out_dir = os.path.join(out_dir, enzyme_id)
    return out_dir


def _apply_gate(store: "libstore.CandidateStore",
                flog: "libfeedback.FailureLog",
                constraints: Dict[str, Any]) -> None:
    """布尔闸: 逐候选 evaluate + apply_gate + 记 failure_log(原地改 store/flog)。"""
    libgate.normalize_constraints(constraints)   # 非法阈值尽早报错
    for r in list(store.records.values()):
        # 续跑: 只处理未终局的 pending / scored; 已过闸/已剔除者不重判。
        if r["status"] not in ("pending", "scored"):
            continue
        res = libgate.evaluate(r["proxy_scores"], constraints)
        store.apply_gate(r["key"], res)
        if not res.passed:
            reasons = ",".join(c.reason for c in res.checks if not c.ok) or "vacuous_fail"
            flog.append(variant_id=r["variant_id"], stage="gate",
                        reason=reasons, proxy_scores=r["proxy_scores"])


def _apply_diverse(store: "libstore.CandidateStore",
                   n_select: int, features: Sequence[str]) -> Dict[str, Any]:
    """多样性采样: 从 passed_gate 挑 n_select, 写 selected(返回 div coverage)。"""
    passed = store.by_status("passed_gate")
    div = libdiverse.select_diverse(passed, n_select, features=features)
    store.select([c["key"] for c in div["selected"]])
    return div


def _apply_export(store: "libstore.CandidateStore", export_cfg: Dict[str, Any],
                  out_dir: str, dry_run: bool,
                  final_selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """导出 selected → evolvepro.tsv(dry_run 只演练不落盘), 返回 export_out。"""
    export_out: Dict[str, Any] = {}
    if export_cfg.get("format", "evolvepro") == "evolvepro":
        if dry_run:
            export_out = {"dry_run": True, "n": len(final_selected),
                          "label_metric": str(export_cfg.get("label_metric", "") or "")}
        else:
            export_out = libexport.export_evolvepro(
                final_selected,
                os.path.join(out_dir, export_cfg.get("out", "evolvepro.tsv")),
                label_metric=str(export_cfg.get("label_metric", "") or ""),
                label_source=str(export_cfg.get("label_source", "proxy") or "proxy"),
                predicts_activity=bool(export_cfg.get("predicts_activity", False)))
    if not dry_run:
        store.mark_exported([r["key"] for r in final_selected])
    return export_out


def run_pipeline(manifest: Dict[str, Any], *,
                 dry_run: bool = False,
                 fresh: bool = False,
                 out_dir: str = "") -> Dict[str, Any]:
    """执行预筛管线(原子: 建一次 store → 顺序跑全部阶段 → 落盘一次), 返回 summary。"""
    enzyme_id = str(manifest.get("enzyme_id", "") or "")
    ms = _resolve_mutationset(manifest.get("mutationset"))

    constraints = manifest.get("constraints", {}) or {}
    libgate.normalize_constraints(constraints)   # 先归一, 非法阈值尽早报错

    diversity = manifest.get("diversity", {}) or {}
    n_select = int(diversity.get("n_select", 30))
    features = tuple(diversity.get("features", ["site", "aa_type"]))

    export_cfg = manifest.get("export", {}) or {}

    out_dir = _resolve_out_dir(manifest, out_dir)
    store_path = os.path.join(out_dir, "candidate_store.json")
    failure_path = os.path.join(out_dir, "failure_log.json")

    # --- 建 store(续跑复用) ---
    if not fresh and os.path.exists(store_path):
        store = libstore.CandidateStore.read_json(store_path)
    else:
        store = libstore.CandidateStore(enzyme_id, ms.template_sequence)
        store.add_mutation_set(ms)

    # --- 打分(可插拔适配器) ---
    scoring_summary = _score_adapter(store, ms, manifest)

    # --- 布尔闸 ---
    flog = libfeedback.FailureLog()
    if not fresh and os.path.exists(failure_path):
        flog = libfeedback.FailureLog.read_json(failure_path)
    _apply_gate(store, flog, constraints)

    n_pass = sum(1 for r in store.records.values()
                 if (r.get("gate") or {}).get("passed"))
    n_rej = len(store.by_status("rejected"))

    # --- 多样性采样(仅从 passed_gate 新挑) ---
    div = _apply_diverse(store, n_select, features)

    # --- 折叠回测(可选, 非 dry_run 才发 API / 落盘) ---
    selected = store.by_status("selected")
    folding_cfg = manifest.get("folding", {}) or {}
    folding_summary: Dict[str, Any] = {"adapter": "", "n_folded": 0}
    if folding_cfg.get("adapter") and not dry_run:
        folding_summary = _fold_adapter(store, ms, folding_cfg, out_dir, selected, flog)

    # --- 第二层精筛(软指标排序 + 二次采样, 可选; 全量接入) ---
    refine_cfg = manifest.get("refine", {}) or {}
    refine_out = _refine_stage(store, selected, refine_cfg, features,
                               int(manifest.get("n_final", len(selected)) or len(selected)))
    final_selected = refine_out["final"]

    # --- 导出(dry_run 不落盘任何文件) ---
    export_out = _apply_export(store, export_cfg, out_dir, dry_run, final_selected)

    # --- 落盘 ---
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        store.write_json(store_path)
        flog.write_json(failure_path)

    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "enzyme_id": enzyme_id,
        "out_dir": out_dir,
        "n_variants": len(store.records),
        "n_scored": sum(1 for r in store.records.values() if r["proxy_scores"]),
        "n_passed_gate": n_pass,
        "n_rejected": n_rej,
        "n_selected": len(final_selected),
        "n_exported": len(store.by_status("exported")),
        "diversity": div["coverage"],
        "scoring": scoring_summary,
        "folding": folding_summary,
        "refine": refine_out,
        "store_path": store_path,
        "failure_log_path": failure_path,
        "export": export_out,
    }


# 预筛管线的五个可独立执行的阶段(命令式 `dockbuddy <stage>` + 声明式 `run` 编排)。
_STAGES = ("score", "gate", "diverse", "fold", "export")


def run_stage(manifest: Dict[str, Any], *, stage: str,
              dry_run: bool = False, fresh: bool = False,
              out_dir: str = "") -> Dict[str, Any]:
    """执行预筛管线的**单阶段**, 供命令式 `dockbuddy <stage>` 与声明式 `run` 编排调用。

    每个阶段从 candidate_store.json 加载(或建)store, 只跑本阶段, 立即落盘(断点粒度
    = 阶段)。阶段间状态经 candidate_store.json 传递: score 推进 scored → gate 推进
    passed_gate/rejected → diverse 推进 selected → fold 回写 proxy_scores → export
    推进 exported。重跑某阶段(不带 --fresh)幂等跳过已推进记录(与 run_pipeline 同口径),
    支撑 GPU 竞价实例回收后的断点续跑。
    """
    if stage not in _STAGES:
        raise ValueError(f"未知 stage: {stage!r} (支持 {sorted(_STAGES)})")
    ms = _resolve_mutationset(manifest.get("mutationset"))
    enzyme_id = str(manifest.get("enzyme_id", "") or "")

    out_dir = _resolve_out_dir(manifest, out_dir)
    store_path = os.path.join(out_dir, "candidate_store.json")
    failure_path = os.path.join(out_dir, "failure_log.json")

    # --- 建 store(续跑复用) ---
    if not fresh and os.path.exists(store_path):
        store = libstore.CandidateStore.read_json(store_path)
    else:
        store = libstore.CandidateStore(enzyme_id, ms.template_sequence)
        store.add_mutation_set(ms)

    flog = libfeedback.FailureLog()
    if not fresh and os.path.exists(failure_path):
        flog = libfeedback.FailureLog.read_json(failure_path)

    result: Dict[str, Any] = {"stage": stage}
    if stage == "score":
        result["scoring"] = _score_adapter(store, ms, manifest)
    elif stage == "gate":
        _apply_gate(store, flog, manifest.get("constraints", {}) or {})
    elif stage == "diverse":
        diversity = manifest.get("diversity", {}) or {}
        result["diversity"] = _apply_diverse(
            store, int(diversity.get("n_select", 30)),
            tuple(diversity.get("features", ["site", "aa_type"])))["coverage"]
    elif stage == "fold":
        folding_cfg = manifest.get("folding", {}) or {}
        if folding_cfg.get("adapter") and not dry_run:
            result["folding"] = _fold_adapter(store, ms, folding_cfg, out_dir,
                                              store.by_status("selected"), flog)
        else:
            result["folding"] = {"adapter": "", "n_folded": 0}
    elif stage == "export":
        selected = store.by_status("selected")
        diversity = manifest.get("diversity", {}) or {}
        features = tuple(diversity.get("features", ["site", "aa_type"]))
        refine_cfg = manifest.get("refine", {}) or {}
        refine_out = _refine_stage(store, selected, refine_cfg, features,
                                   int(manifest.get("n_final", len(selected))
                                       or len(selected)))
        result["refine"] = refine_out
        result["export"] = _apply_export(store, manifest.get("export", {}) or {},
                                         out_dir, dry_run, refine_out["final"])

    # --- 落盘(每个阶段独立落盘, 支撑断点续跑) ---
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        store.write_json(store_path)
        flog.write_json(failure_path)

    result.update({
        "ok": True,
        "dry_run": bool(dry_run),
        "enzyme_id": enzyme_id,
        "out_dir": out_dir,
        "n_variants": len(store.records),
        "n_scored": sum(1 for r in store.records.values() if r["proxy_scores"]),
        "n_passed_gate": sum(1 for r in store.records.values()
                             if (r.get("gate") or {}).get("passed")),
        "n_rejected": len(store.by_status("rejected")),
        "n_selected": len(store.by_status("selected")),
        "n_exported": len(store.by_status("exported")),
        "store_path": store_path,
        "failure_log_path": failure_path,
    })
    return result


__all__ = ["run_pipeline", "run_stage", "DEFAULT_OUT_DIR"]
