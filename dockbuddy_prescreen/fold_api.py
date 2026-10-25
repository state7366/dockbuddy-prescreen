# -*- coding: utf-8 -*-
"""
libfold_api.py —— 结构预测适配器(ESMFold 双通道: 公共 API / GPU 本地, 2026-10-09)
===================================================================================
定位: 预筛管线「单体回测」(阶段④)的结构折叠适配器。ESMFold 提供**两种 transport**:

  1. api   (公共 REST API, 无 GPU):  POST https://api.esmatlas.com/foldSequence/v1/pdb/
           body = 单链氨基酸序列(纯文本), 返回完整 PDB 文本。免费、无鉴权、单链、
           约 ≤400 aa、pLDDT 写在 B 因子列(与 AlphaFold 口径一致)。适合无 GPU 的
           开发机 / CI / 轻量验证。
  2. local (GPU 服务器本地部署):     `esm.pretrained.esmfold_v1()`(fair-esm) 本地
           推理, 无需外网 API。适合自建 GPU 实例/快机、序列超长(>400aa)、隐私序列、
           或公共 API 不稳定时。依赖 torch + fair-esm[esmfold] + openfold + 权重。

设计决策(记录备查):
  - api 通道零依赖: 只用标准库 urllib(与 fetch.py 同口径), 不引入 requests。
  - local 通道懒加载: 模型首次调用才加载并缓存(模块级单例), 避免重复加载 3B 权重。
  - 代理(仅 api): 显式 proxy 参数 > DOCK_PROXY 环境变量 > 直连。
  - 不预测活性: 返回的是结构坐标 + pLDDT, 是「结构回测」输入, 不是功能/活性打分。
  - 失败即抛 RuntimeError(带状态/原因), 不静默返回半截结构。
"""
from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request

LOG = logging.getLogger("dockpipeline")

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

_ESMFOLD_LOCAL_HINT = (
    "ESMFold 本地部署需要 torch + fair-esm[esmfold] + openfold + esmfold_3B 权重 "
    "(GPU 显存 ≥16GB)。安装: pip install 'fair-esm[esmfold]' + "
    "pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git"
    "@4b41059694619831a7db195b7e0988fc4ff3a307'; 权重由 torch hub 首次加载时自动"
    "下载(受 TORCH_HOME 控制)。本机未部署时请用 transport=api 或改 GPU 服务器部署。"
)

# 模块级模型缓存(懒加载单例): 首次 esmfold_pdb_local 时加载, 之后复用。
_ESMFOLD_MODEL = None


def _default_proxy() -> str:
    return os.environ.get("DOCK_PROXY", "")


def _post(url: str, payload: bytes, proxy: str, timeout: int) -> str:
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "text/plain; charset=utf-8")
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API HTTP {e.code} ({url})") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"API 请求失败 ({url}): {e}") from e


def esmfold_pdb(sequence: str, proxy: str = "", timeout: int = 300) -> str:
    """用 ESMFold 公开 API 折叠单链序列, 返回 PDB 文本。

    参数:
      sequence: 单链氨基酸序列(1-letter; 内部忽略空白)。
      proxy: HTTP/HTTPS 代理(缺省取 DOCK_PROXY 环境变量)。
      timeout: 秒(缺省 300; 长序列/网络慢时需更长)。
    返回: PDB 文本。失败抛 RuntimeError(带状态码/原因)。
    """
    seq = "".join(sequence.split())
    if not seq:
        raise ValueError("esmfold_pdb: 空序列")
    body = _post(ESMFOLD_URL, seq.encode("ascii", "ignore"), proxy or _default_proxy(), timeout)
    head = body.lstrip()
    if not head.startswith(("HEADER", "ATOM", "TITLE")):
        raise RuntimeError(f"ESMFold API 返回非 PDB 内容: {head[:80]!r}")
    return body


def esmfold_to_file(sequence: str, out_pdb: str, proxy: str = "",
                    timeout: int = 300) -> str:
    """折叠并落盘 PDB, 返回绝对路径。"""
    pdb = esmfold_pdb(sequence, proxy=proxy, timeout=timeout)
    out_pdb = os.path.abspath(out_pdb)
    os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
    with open(out_pdb, "w", encoding="utf-8") as f:
        f.write(pdb)
    return out_pdb


def esmfold_pdb_local(sequence: str, device: str = "auto",
                      chunk_size: int = 0) -> str:
    """用本地 ESMFold(fair-esm) GPU 推理折叠单链序列, 返回 PDB 文本。

    参数:
      sequence: 单链氨基酸序列(1-letter; 内部忽略空白)。
      device: 'auto'/'cuda'/'cpu'(缺省自动探测, 见 common.resolve_device)。
      chunk_size: axial-attention 分块大小(>0 时启用, 降低显存占用; 0 关闭)。
        长序列(<~700aa)显存吃紧时可设 128/64; 分块越细越慢。
    返回: PDB 文本(pLDDT 写在 B 因子列, 与公共 API 口径一致)。
    失败: 依赖/权重缺失抛 RuntimeError(带部署指引)。
    """
    global _ESMFOLD_MODEL
    try:
        import torch  # noqa: F401
        import esm  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_ESMFOLD_LOCAL_HINT) from e

    from dockbuddy_common import resolve_device
    dev = resolve_device(str(device or "auto"))

    if _ESMFOLD_MODEL is None:
        try:
            model = esm.pretrained.esmfold_v1()
        except Exception as e:  # noqa: BLE001 —— openfold/权重缺失等, 转清晰报错
            raise RuntimeError(f"{_ESMFOLD_LOCAL_HINT} (加载失败: {e})") from e
        model = model.eval()
        if dev == "cuda":
            model = model.cuda()
        if chunk_size and chunk_size > 0:
            model.set_chunk_size(int(chunk_size))
        _ESMFOLD_MODEL = model

    seq = "".join(str(sequence).split())
    if not seq:
        raise ValueError("esmfold_pdb_local: 空序列")
    with torch.no_grad():
        pdb = _ESMFOLD_MODEL.infer_pdb(seq)
    # 2026-10-10 修正: esmfold_v1().infer_pdb() 的输出首行是 "PARENT N/A"(父结构
    # 标注, 无父结构时为 N/A), 之后才是 ATOM 行。旧校验只认 HEADER/ATOM/TITLE,
    # 会把合法的折叠结果误判为非 PDB。加 "PARENT" 前缀, 且以是否含 ATOM 行为准。
    text = str(pdb)
    if not text.lstrip().startswith(("HEADER", "ATOM", "TITLE", "PARENT", "REMARK")):
        raise RuntimeError(f"ESMFold 本地推理返回非 PDB 内容: {text[:80]!r}")
    if "ATOM" not in text:
        raise RuntimeError(f"ESMFold 本地推理结果无 ATOM 坐标: {text[:80]!r}")
    return text


def _find_colabfold() -> str:
    import shutil
    return shutil.which("colabfold_batch") or ""


_COLABFOLD_HINT = ("ColabFold 需 colabfold_batch 二进制 + 权重(部署见 deploy/)。"
                   "装好 colabfold 后: pip install 'colabfold[alphafold]' 或 "
                   "conda install -c conda-forge -c bioconda colabfold; "
                   "权重首次运行自动下载, 或设 COLABFOLD_BIN 指向二进制。")


def colabfold_to_file(sequence: str, out_pdb: str, colabfold_bin: str = "",
                      msa_mode: str = "mmseqs2_uniref_env", tmp_dir: str = "",
                      timeout: int = 3600) -> str:
    """用 ColabFold(colabfold_batch) 折叠单链序列, 落盘 PDB, 返回绝对路径。

    参数:
      sequence: 单链 1-letter 序列。
      out_pdb: 输出 PDB 路径(取 rank_1 未松弛模型)。
      colabfold_bin: colabfold_batch 路径(缺省 which / COLABFOLD_BIN)。
      msa_mode: mmseqs2_uniref_env(缺省) / single_sequence / unpaired_paired。
      tmp_dir: 工作目录(缺省 out_pdb 同目录的临时子目录)。
      timeout: 子进程超时(秒)。
    """
    import glob
    import shutil
    import subprocess
    import tempfile

    bin_path = colabfold_bin or os.environ.get("COLABFOLD_BIN", "") or _find_colabfold()
    if not bin_path:
        raise RuntimeError(_COLABFOLD_HINT)
    seq = "".join(str(sequence).split())
    if not seq:
        raise ValueError("colabfold_to_file: 空序列")

    out_pdb = os.path.abspath(out_pdb)
    work = tmp_dir or os.path.join(os.path.dirname(out_pdb) or ".", "_colabfold_tmp")
    os.makedirs(work, exist_ok=True)
    fasta = os.path.join(work, "query.fasta")
    with open(fasta, "w", encoding="utf-8") as f:
        f.write(f">query\n{seq}\n")
    outdir = os.path.join(work, "out")
    os.makedirs(outdir, exist_ok=True)

    cmd = [bin_path, fasta, outdir, "--msa-mode", msa_mode]
    LOG.info("ColabFold: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError("ColabFold 失败:\n" + (proc.stderr or proc.stdout)[-2000:])

    # 结果: outdir/<query>_unrelaxed_rank_001_*.pdb 或 ranked_0.pdb
    cands = sorted(glob.glob(os.path.join(outdir, "*rank_001*.pdb"))) or \
        sorted(glob.glob(os.path.join(outdir, "*ranked_0*.pdb")))
    if not cands:
        raise RuntimeError(f"ColabFold 未产出 PDB(outdir={outdir})")
    src = cands[0]
    os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
    shutil.copyfile(src, out_pdb)
    return out_pdb


__all__ = ["esmfold_pdb", "esmfold_pdb_local", "esmfold_to_file", "colabfold_to_file",
           "ESMFOLD_URL"]
