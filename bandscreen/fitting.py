# -*- coding: utf-8 -*-
"""CBM/VBM 带边的固定顶点二次拟合（逐段标定版）。

核心步骤（fit_edge_sides）：
    1. 包络极值定带边点，像素自适应容差内吸附到最近的段边界节点；
    2. 左右拟合窗口在带边所在段的边界处截断（防相邻段分支混入）；
    3. 顶点能量 E0 单侧去偏重估（消除包络顺序统计量偏差）；
    4. 左右两侧分别做固定顶点二次拟合 E-E0=A(k-k0)²，曲率 C=2A，
       m*/m0 = 7.62/|C|；附近线性/可靠性诊断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .calibrate import PiecewiseKmap
from .common import HBAR2_OVER_M0_EV_A2


@dataclass
class SideFit:
    side: str
    n_points: int
    k_span_Ainv: Optional[float]
    energy_span_eV: Optional[float]
    curvature_eVA2: Optional[float]
    m_eff_m0_from_image: Optional[float]
    sign_ok: bool
    quad_rmse_eV: Optional[float]
    linear_rmse_eV: Optional[float]
    quadratic_energy_span_eV: Optional[float]
    near_linear_flag: bool
    curvature_reliable: bool
    linear_slope_eVA: Optional[float]
    mean_abs_slope_eVA: Optional[float]
    max_abs_slope_eVA: Optional[float]
    side_points: pd.DataFrame
    quad_curve: pd.DataFrame
    line_curve: pd.DataFrame


@dataclass
class EdgeResult:
    kind: str
    edge_E: float
    edge_k_norm: float
    edge_x: float
    edge_y: float
    left: SideFit
    right: SideFit
    envelope: pd.DataFrame
    local_points: pd.DataFrame
    edge_k_norm_raw: float = float("nan")
    vertex_snapped: bool = False


def _empty_side(side: str) -> SideFit:
    return SideFit(
        side=side,
        n_points=0,
        k_span_Ainv=None,
        energy_span_eV=None,
        curvature_eVA2=None,
        m_eff_m0_from_image=None,
        sign_ok=False,
        quad_rmse_eV=None,
        linear_rmse_eV=None,
        quadratic_energy_span_eV=None,
        near_linear_flag=True,
        curvature_reliable=False,
        linear_slope_eVA=None,
        mean_abs_slope_eVA=None,
        max_abs_slope_eVA=None,
        side_points=pd.DataFrame(),
        quad_curve=pd.DataFrame(),
        line_curve=pd.DataFrame(),
    )


def _fit_side_fixed_vertex(
    side_df: pd.DataFrame,
    side: str,
    kind: str,
    q0: float,
    E0: float,
    kmap: PiecewiseKmap,
    min_side_points: int,
    rmse_threshold_eV: float,
    min_quadratic_span_eV: float,
    linear_preference_tol: float,
) -> SideFit:
    if side_df is None or side_df.empty or len(side_df) < min_side_points:
        return _empty_side(side)

    df = side_df.sort_values("k_norm").copy()
    # 逐段标定：q -> k(Å^-1) 用分段线性映射，拟合只依赖带边所在段的局部尺度。
    k0_abs = float(kmap.k(q0))
    dk = kmap.k(df["k_norm"].to_numpy(dtype=float)) - k0_abs  # Å^-1
    dE = df["E"].to_numpy(dtype=float) - E0
    if np.max(np.abs(dk)) <= 1e-12:
        return _empty_side(side)

    # Fixed-vertex quadratic: E - E0 = A * (k-k0)^2
    z = dk ** 2
    denom = float(np.sum(z * z))
    if denom <= 1e-20:
        return _empty_side(side)
    A = float(np.sum(z * dE) / denom)  # eV·Å²
    dE_quad = A * z
    curvature = float(2.0 * A)  # eV·Å²
    quad_rmse = float(np.sqrt(np.mean((dE_quad - dE) ** 2)))

    # Through-vertex line for near-linear diagnosis and auxiliary slope.
    denom_line = float(np.sum(dk * dk))
    if denom_line > 1e-20:
        linear_slope = float(np.sum(dk * dE) / denom_line)  # eV·Å
        dE_line = linear_slope * dk
        linear_rmse = float(np.sqrt(np.mean((dE_line - dE) ** 2)))
    else:
        linear_slope = None
        linear_rmse = None

    # Local finite-difference slopes as auxiliary dispersion descriptors.
    if len(df) >= 2:
        E_sorted = df["E"].to_numpy(dtype=float)
        dks = np.diff(dk)
        dEs = np.diff(E_sorted)
        good = np.abs(dks) > 1e-12
        if np.any(good):
            slopes = np.abs(dEs[good] / dks[good])  # eV·Å
            mean_abs_slope = float(np.mean(slopes))
            max_abs_slope = float(np.max(slopes))
        else:
            mean_abs_slope = None
            max_abs_slope = None
    else:
        mean_abs_slope = None
        max_abs_slope = None

    sign_ok = bool(curvature > 0) if kind == "cbm" else bool(curvature < 0)
    if sign_ok and abs(curvature) > 1e-12:
        m_eff = HBAR2_OVER_M0_EV_A2 / abs(curvature)
    else:
        m_eff = None

    k_span = float(np.max(dk) - np.min(dk))
    E_span = float(np.max(df["E"]) - np.min(df["E"]))
    quadratic_span = float(abs(A) * (np.max(np.abs(dk)) ** 2))

    # 如果二次项贡献低于能量噪声，或线性模型并不比二次模型差，则标记为近似线性。
    near_linear = False
    if quadratic_span < min_quadratic_span_eV:
        near_linear = True
    if linear_rmse is not None and linear_rmse <= quad_rmse * (1.0 + linear_preference_tol):
        near_linear = True

    reliable = bool(len(df) >= min_side_points and sign_ok and quad_rmse <= rmse_threshold_eV and not near_linear)

    # Curves for debug.
    dk_min, dk_max = float(np.min(dk)), float(np.max(dk))
    xs = np.linspace(dk_min, dk_max, 100)
    qs = kmap.q(k0_abs + xs)
    quad_Es = E0 + A * xs ** 2
    quad_curve = pd.DataFrame({"k_norm": qs, "E": quad_Es})
    if linear_slope is not None:
        line_curve = pd.DataFrame({"k_norm": qs, "E": E0 + linear_slope * xs})
    else:
        line_curve = pd.DataFrame(columns=["k_norm", "E"])

    return SideFit(
        side=side,
        n_points=int(len(df)),
        k_span_Ainv=k_span,
        energy_span_eV=E_span,
        curvature_eVA2=curvature,
        m_eff_m0_from_image=m_eff,
        sign_ok=sign_ok,
        quad_rmse_eV=quad_rmse,
        linear_rmse_eV=linear_rmse,
        quadratic_energy_span_eV=quadratic_span,
        near_linear_flag=near_linear,
        curvature_reliable=reliable,
        linear_slope_eVA=linear_slope,
        mean_abs_slope_eVA=mean_abs_slope,
        max_abs_slope_eVA=max_abs_slope,
        side_points=df,
        quad_curve=quad_curve,
        line_curve=line_curve,
    )


def fit_edge_sides(
    env: pd.DataFrame,
    kind: str,
    local_width: float,
    energy_window: float,
    kmap: PiecewiseKmap,
    min_side_points: int,
    rmse_threshold_eV: float,
    min_quadratic_span_eV: float,
    linear_preference_tol: float,
    vertex_snap_cols: float = 3.5,
) -> Optional[EdgeResult]:
    if env.empty:
        return None
    if kind == "cbm":
        idx = env["E"].idxmin()
    elif kind == "vbm":
        idx = env["E"].idxmax()
    else:
        raise ValueError("kind must be cbm or vbm")
    edge_E = float(env.loc[idx, "E"])
    edge_k = float(env.loc[idx, "k_norm"])

    # 带边极值点落在段边界 snap 容差内时吸附到边界节点。带边位于
    # 高对称点是常态；分隔线列被剔除/线宽会使包络顶点偏移 1-3 像素，
    # 顶点偏移经固定顶点拟合会放大成左右不对称的曲率偏差，吸附消除之。
    # 容差按像素自适应（vertex_snap_cols 个 k 列，默认 3.5；分隔线列
    # 被显式剔除的场景——如任意图分析器——应传更大值），避免高分辨率
    # 图上把真实偏离节点的顶点错误拽到高对称点。
    q_cols = np.unique(env["k_norm"].to_numpy(dtype=float))
    col_dq = float(np.median(np.diff(q_cols))) if q_cols.size >= 2 else 0.002
    vertex_snap_tol = float(vertex_snap_cols) * col_dq
    edge_k_raw = edge_k
    vertex_snapped = False
    snap_nodes = np.asarray(kmap.q_nodes, dtype=float)
    if snap_nodes.size:
        j = int(np.argmin(np.abs(snap_nodes - edge_k)))
        if abs(float(snap_nodes[j]) - edge_k) <= vertex_snap_tol and abs(float(snap_nodes[j]) - edge_k) > 1e-12:
            edge_k = float(snap_nodes[j])
            vertex_snapped = True

    if kind == "cbm":
        local = env[(np.abs(env["k_norm"] - edge_k) <= local_width) & (env["E"] <= edge_E + energy_window)].copy()
    else:
        local = env[(np.abs(env["k_norm"] - edge_k) <= local_width) & (env["E"] >= edge_E - energy_window)].copy()

    # 左右拟合窗口在带边所在段的边界处截断，避免把方向在高对称点
    # 转折的相邻段（不同晶向的分支）混进同一条固定顶点抛物线。
    # 截断边界只采用距带边至少 ~10 列的节点：更近的节点若充当边界会
    # 产生几个像素宽的碎片拟合窗口（k 跨度过小，曲率数值不稳定），
    # 此时跳过该节点、窗口延伸到下一节点。
    nodes = np.asarray(kmap.q_nodes, dtype=float)
    snap = max(vertex_snap_tol, 10.0 * col_dq, 1e-6)
    lower_nodes = nodes[nodes < edge_k - snap]
    upper_nodes = nodes[nodes > edge_k + snap]
    left_bound = float(lower_nodes.max()) if lower_nodes.size else -np.inf
    right_bound = float(upper_nodes.min()) if upper_nodes.size else np.inf

    left_df = local[(local["k_norm"] < edge_k - 1e-9) & (local["k_norm"] >= left_bound)].copy()
    right_df = local[(local["k_norm"] > edge_k + 1e-9) & (local["k_norm"] <= right_bound)].copy()

    # 顶点能量去偏。包络的极值是"逐列取极值再全局取极值"的顺序
    # 统计量，被像素量化噪声系统性推向窗口外侧（约半像素能量），固定
    # 顶点拟合会把该偏移放大成近顶点点的曲率畸变（低分辨率图像显著）。
    # 用共享顶点能量、左右曲率独立的线性最小二乘重估 E0：
    #   E_i = E0 + A_side * (k_i - k0)^2
    k0_abs_ref = float(kmap.k(edge_k))
    rows_z: List[Tuple[float, float, int]] = []
    for side_idx, sdf in ((0, left_df), (1, right_df)):
        if sdf is not None and not sdf.empty:
            dk_s = kmap.k(sdf["k_norm"].to_numpy(dtype=float)) - k0_abs_ref
            for z, E in zip(dk_s ** 2, sdf["E"].to_numpy(dtype=float)):
                rows_z.append((float(z), float(E), side_idx))
    n_sides_present = len({c for _, _, c in rows_z})
    if len(rows_z) >= 3 + n_sides_present and n_sides_present >= 1:
        M = np.zeros((len(rows_z), 1 + n_sides_present))
        yv = np.zeros(len(rows_z))
        side_col = {c: i + 1 for i, c in enumerate(sorted({c for _, _, c in rows_z}))}
        for i, (z, E, c) in enumerate(rows_z):
            M[i, 0] = 1.0
            M[i, side_col[c]] = z
            yv[i] = E
        try:
            sol, *_ = np.linalg.lstsq(M, yv, rcond=None)
            E0_ref = float(sol[0])
            # 接受窗口：该重估要纠正的是约半个像素的顺序统计量偏差，
            # 因此只接受不超过 ~2 像素能量、且方向正确的修正（CBM 的
            # 包络极值只会偏低、VBM 只会偏高）。对称宽窗口会让近线性
            # 带边被抬出假顶点、绕过 near_linear 保护。
            uniq_E = np.unique(local["E"].to_numpy(dtype=float))
            dE_q = float(np.median(np.diff(uniq_E))) if uniq_E.size >= 3 else 0.005
            eps_px = max(4.0 * dE_q, 1e-6)
            if kind == "cbm" and edge_E <= E0_ref <= edge_E + eps_px:
                edge_E = E0_ref
            elif kind == "vbm" and edge_E - eps_px <= E0_ref <= edge_E:
                edge_E = E0_ref
        except Exception:
            pass

    left_fit = _fit_side_fixed_vertex(left_df, "left", kind, edge_k, edge_E, kmap, min_side_points, rmse_threshold_eV, min_quadratic_span_eV, linear_preference_tol)
    right_fit = _fit_side_fixed_vertex(right_df, "right", kind, edge_k, edge_E, kmap, min_side_points, rmse_threshold_eV, min_quadratic_span_eV, linear_preference_tol)

    return EdgeResult(
        kind=kind,
        edge_E=edge_E,
        edge_k_norm=edge_k,
        edge_x=float(env.loc[idx, "x"]),
        edge_y=float(env.loc[idx, "y"]),
        left=left_fit,
        right=right_fit,
        envelope=env,
        local_points=local,
        edge_k_norm_raw=edge_k_raw,
        vertex_snapped=vertex_snapped,
    )
