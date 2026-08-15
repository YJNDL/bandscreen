#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kpath_segments.py

由晶体结构计算高对称 k 路径各段的物理长度（Å^-1）。

这是"任意能带图判断"管线的结构侧输入模块：图像端只提供高对称点
标签序列和它们的像素位置；本模块负责把 (结构, 标签对) 换算成该段
的倒空间长度，供逐段像素→Å^-1 标定使用。

三个入口：
    segments_from_structure(structure_file, labels)   POSCAR/CIF -> 段长列表
    segments_from_mp(formula_or_mpid, labels)         化学式/mp-ID -> 段长列表（需 MP_API_KEY）
    pair_length(structure, label1, label2)            单段长度

标签归一化：接受 G / GAMMA / Γ / \\Gamma / 希腊同形字（Χ Μ Κ）等写法。

自测（fcc Si 的 Γ-X = 2π/a 等闭式解验证）：
    python kpath_segments.py --selftest
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 希腊同形字 -> 拉丁；OCR 端和文献标签常混用
_HOMOGLYPHS = {
    "Χ": "X", "Μ": "M", "Κ": "K", "Α": "A", "Β": "B", "Ε": "E",
    "Ζ": "Z", "Η": "H", "Ι": "I", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Υ": "Y",
}

# pymatgen HighSymmKpath 使用 LaTeX 风格 "\Gamma"；QuantumATK route 用 "G"
_GAMMA_ALIASES = {"G", "GAMMA", "Γ", r"\GAMMA", "\\GAMMA", "$\\GAMMA$"}


def normalize_label(label: str) -> str:
    """把任意来源的高对称点标签归一化为统一键（GAMMA / X / M / K_1 ...）。"""
    s = str(label).strip()
    s = s.replace("$", "").replace("{", "").replace("}", "")
    for a, b in _HOMOGLYPHS.items():
        s = s.replace(a, b)
    up = s.upper().lstrip("\\")
    if up in {a.lstrip("\\$") for a in _GAMMA_ALIASES} or up == "GAMMA":
        return "GAMMA"
    return up


def _load_structure(structure_file: str):
    from pymatgen.core import Structure
    return Structure.from_file(structure_file)


def _kpath_frac_coords(structure, path_type: str = "setyawan_curtarolo") -> Dict[str, np.ndarray]:
    """
    标准路径高对称点分数坐标表（相对标准化原胞的倒格矢）。

    返回 {归一化标签: 分数坐标}；同时返回的第二个值是标准化原胞
    （段长必须用它的倒格子计算，直接用常规胞会算错 fcc/bcc）。
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.symmetry.bandstructure import HighSymmKpath

    prim = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
    kpath = HighSymmKpath(prim, path_type=path_type)
    coords = {
        normalize_label(lbl): np.asarray(frac, dtype=float)
        for lbl, frac in kpath.kpath["kpoints"].items()
    }
    return coords, prim


def pair_length(structure, label1: str, label2: str, path_type: str = "setyawan_curtarolo") -> float:
    """两个高对称点之间的倒空间距离（Å^-1）。reciprocal_lattice 已含 2π。"""
    coords, prim = _kpath_frac_coords(structure, path_type=path_type)
    l1, l2 = normalize_label(label1), normalize_label(label2)
    for l in (l1, l2):
        if l not in coords:
            raise KeyError(
                f"Label {l!r} not in standard k-path of this structure; "
                f"available: {sorted(coords)}"
            )
    cart = prim.lattice.reciprocal_lattice.get_cartesian_coords(coords[l2] - coords[l1])
    return float(np.linalg.norm(cart))


def segments_from_structure(
    structure_file: str,
    labels: Sequence[str],
    path_type: str = "setyawan_curtarolo",
) -> List[float]:
    """
    给定结构文件与图上识别出的标签序列，返回相邻标签对的段长列表（Å^-1）。

    labels 长度为 N，返回长度 N-1。不连续路径的断点（如 "X|K"）应在
    调用前拆开，两侧分别作为独立标签传入——断点两侧像素重合、段长为 0，
    调用方可传 "X|K" 原样，本函数对含 "|" 的标签取 "|" 前半段作为段终点、
    后半段作为下一段起点。
    """
    structure = _load_structure(structure_file)
    coords, prim = _kpath_frac_coords(structure, path_type=path_type)
    return _pairwise_segment_lengths(coords, prim.lattice.reciprocal_lattice, labels)


def _pairwise_segment_lengths(coords: Dict[str, np.ndarray], B, labels: Sequence[str]) -> List[float]:
    """相邻标签对的段长（Å^-1），含 "X|K" 断点标签的统一处理。"""

    def coord_of(raw: str, take_end: bool) -> np.ndarray:
        s = str(raw)
        if "|" in s:
            s = s.split("|")[0] if take_end else s.split("|")[-1]
        l = normalize_label(s)
        if l not in coords:
            raise KeyError(f"Label {l!r} not in standard k-path; available: {sorted(coords)}")
        return coords[l]

    out: List[float] = []
    for i in range(len(labels) - 1):
        k1 = coord_of(labels[i], take_end=False)
        k2 = coord_of(labels[i + 1], take_end=True)
        cart = B.get_cartesian_coords(k2 - k1)
        out.append(float(np.linalg.norm(cart)))
    return out


def segments_from_mp(
    formula_or_mpid: str,
    labels: Sequence[str],
    path_type: str = "setyawan_curtarolo",
) -> List[float]:
    """
    只有化学式/mp-ID 时：从 Materials Project 取最稳定相结构再算段长。
    需要环境变量 MP_API_KEY（https://materialsproject.org/api 免费注册）。
    """
    from mp_api.client import MPRester

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY not set; register at https://materialsproject.org/api")
    with MPRester(api_key) as mpr:
        if str(formula_or_mpid).startswith("mp-"):
            docs = mpr.materials.summary.search(
                material_ids=[formula_or_mpid], fields=["material_id", "structure"]
            )
        else:
            docs = mpr.materials.summary.search(
                formula=formula_or_mpid,
                fields=["material_id", "structure", "energy_above_hull"],
            )
            docs = sorted(docs, key=lambda d: (d.energy_above_hull is None, d.energy_above_hull))
        if not docs:
            raise RuntimeError(f"No Materials Project entry for {formula_or_mpid!r}")
    structure = docs[0].structure
    coords, prim = _kpath_frac_coords(structure, path_type=path_type)
    return _pairwise_segment_lengths(coords, prim.lattice.reciprocal_lattice, labels)


def _selftest() -> int:
    """闭式解验证：fcc Si 与 hex 结构的标准段长。"""
    from pymatgen.core import Lattice, Structure

    ok = True

    a = 5.431
    si = Structure(
        Lattice.cubic(a), ["Si"] * 8,
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0], [0.75, 0.75, 0.25],
         [0.5, 0, 0.5], [0.75, 0.25, 0.75], [0, 0.5, 0.5], [0.25, 0.75, 0.75]],
    )
    gx = pair_length(si, "Γ", "X")
    expect = 2 * np.pi / a
    err = abs(gx - expect) / expect
    print(f"fcc Si  Γ-X = {gx:.6f} Å^-1, 2π/a = {expect:.6f}, rel err = {err:.2e}")
    ok &= err < 1e-6

    gl = pair_length(si, "GAMMA", "L")
    expect_l = np.sqrt(3) * np.pi / a
    err_l = abs(gl - expect_l) / expect_l
    print(f"fcc Si  Γ-L = {gl:.6f} Å^-1, √3π/a = {expect_l:.6f}, rel err = {err_l:.2e}")
    ok &= err_l < 1e-6

    a_h, c_h = 3.19, 5.19
    gan = Structure(
        Lattice.hexagonal(a_h, c_h), ["Ga", "Ga", "N", "N"],
        [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.375], [2 / 3, 1 / 3, 0.875]],
    )
    gm = pair_length(gan, "G", "M")
    expect_m = 2 * np.pi / (np.sqrt(3) * a_h)
    err_m = abs(gm - expect_m) / expect_m
    print(f"hex GaN Γ-M = {gm:.6f} Å^-1, 2π/(√3a) = {expect_m:.6f}, rel err = {err_m:.2e}")
    ok &= err_m < 1e-4

    try:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "POSCAR")
            si.to(filename=p, fmt="poscar")
            lens = segments_from_structure(p, ["Γ", "X", "W"])
        print(f"segments Γ-X-W = {['%.6f' % v for v in lens]} (expect X-W = π/a = {np.pi / a:.6f})")
        ok &= abs(lens[0] - expect) / expect < 1e-6
        ok &= abs(lens[1] - np.pi / a) / (np.pi / a) < 1e-6
    except Exception as exc:
        print(f"segments_from_structure selftest failed: {exc}")
        ok = False

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
