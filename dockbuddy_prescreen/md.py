# -*- coding: utf-8 -*-
"""prescreen.md —— OpenMM 动力学(能量最小化 + 短 MD)核心(2026-10-11 新增)
===================================================================================
定位: 环节 8「动力学/柔性(MD)」的第一个模型 —— 用 OpenMM(本机已装 openmm 8.6.1 +
pdbfixer)对结构做能量最小化 + 短 MD, 用于:
  1. 突变体结构的能量弛豫验证(最小化后能量下降幅度);
  2. apo MD 空腔时间序列(验 F205G +23.2% 扩口袋是否超空腔自涨落, 项目待办)。

诚实边界:
  - 本模块用 implicit solvent(GBSA-OBC), 非显式水; 短 MD(ps~ns)仅验证动力学
    可行性 + 局部弛豫, 不足以采样大构象变化(需 μs 级 explicit 水 MD)。
  - 空腔时间序列的口径需与 pocket_tunnel 正式口径(Ser OG 球心)一致, 不混用。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def minimize(pdb_path: str, chain: str = "A", out_pdb: str = "",
             platform: str = "CPU", max_iterations: int = 0) -> Dict[str, Any]:
    """PDBFixer 修复 + OpenMM 能量最小化(implicit GBSA-OBC)。

    返回 {energy_before, energy_after, out_pdb, chain}。
    """
    import pdbfixer
    from openmm import app as mmapp
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm.app import CutoffNonPeriodic, Modeller, PDBFile, Simulation
    from openmm.unit import kelvin, picosecond, picoseconds

    fixer = pdbfixer.PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    forcefield = mmapp.ForceField("amber14-all.xml", "implicit/obc2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=CutoffNonPeriodic,
                                     constraints=mmapp.HBonds)
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond,
                                          0.002 * picoseconds)
    platform = {"GPU": "CUDA", "gpu": "CUDA"}.get(platform, platform)
    plat = Platform.getPlatformByName(platform)
    sim = Simulation(modeller.topology, system, integrator, plat)
    sim.context.setPositions(modeller.positions)

    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    sim.minimizeEnergy(maxIterations=max_iterations if max_iterations else 2000)
    e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()

    out = out_pdb or (pdb_path.rsplit(".", 1)[0] + "_minimized.pdb")
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f)

    return {"energy_before": float(e0.value_in_unit(e0.unit)),
            "energy_after": float(e1.value_in_unit(e1.unit)),
            "energy_unit": "kJ/mol", "out_pdb": out}


def short_md(pdb_path: str, chain: str = "A", steps: int = 5000,
             platform: str = "CPU", out_dcd: str = "", report_interval: int = 100,
             seed: int = 42) -> Dict[str, Any]:
    """能量最小化 + 短 MD(Langevin, implicit GBSA-OBC), 记录势能时间序列。

    返回 {energy_series: [(step, pe_kjmol)], out_dcd, steps}。
    """
    import pdbfixer
    from openmm import app as mmapp
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm.app import CutoffNonPeriodic, Modeller, PDBFile, Simulation, DCDReporter
    from openmm.unit import kelvin, picosecond, picoseconds

    fixer = pdbfixer.PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    forcefield = mmapp.ForceField("amber14-all.xml", "implicit/obc2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=CutoffNonPeriodic,
                                     constraints=mmapp.HBonds)
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond,
                                          0.002 * picoseconds)
    integrator.setRandomNumberSeed(seed)
    platform = {"GPU": "CUDA", "gpu": "CUDA"}.get(platform, platform)
    plat = Platform.getPlatformByName(platform)
    sim = Simulation(modeller.topology, system, integrator, plat)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=2000)

    dcd = out_dcd or (pdb_path.rsplit(".", 1)[0] + "_md.dcd")
    if os.path.dirname(dcd):
        os.makedirs(os.path.dirname(dcd), exist_ok=True)
    sim.reporters.append(DCDReporter(dcd, report_interval))

    series = []
    for i in range(0, steps, report_interval):
        sim.step(report_interval)
        pe = sim.context.getState(getEnergy=True).getPotentialEnergy()
        series.append((i + report_interval, float(pe.value_in_unit(pe.unit))))

    return {"energy_series": series, "out_dcd": dcd, "steps": steps,
            "energy_unit": "kJ/mol"}


__all__ = ["minimize", "short_md"]
