#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_band_image.py

任意文献风格能带图的单图判断工具（"随便一张能带图都可以判断"）。

流程：
    1. bandscreen.axes_ocr.analyze_axes：图框检测 + y 轴能量 OCR 标定
       + x 轴高对称点标签识别与像素锚点配对（全部来自图像本身）；
    2. bandscreen.kpath：由晶体结构（POSCAR/CIF 或 Materials Project）
       计算相邻标签对的物理段长（Å^-1）——唯一的声明外部输入；
    3. 复用 bandscreen 的像素提取 / CBM-VBM 包络 / 逐段标定固定顶点
       二次拟合，输出带隙、直接/间接、方向分辨曲率与图像估算有效质量。

分层降级：
    - 无结构信息：仍输出带隙(eV)与直接/间接判断（只需 y 轴标定），
      曲率以无量纲单位（eV/归一化k²）给出，m* 标记不可用；
    - y 轴 OCR 失败：需要 --emin/--emax 手工给定能量窗口。

示例：
    python analyze_band_image.py --image fig3b.png --structure POSCAR
    python analyze_band_image.py --image fig.png --mp mp-149
    python analyze_band_image.py --image fig.png --structure a.cif \
        --labels "G,X,W,L,G" --mode dark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import bandscreen as bs
from bandscreen import axes_ocr as axes_mod
from bandscreen import kpath as kseg

CROP_INSET = 3


def _merge_discontinuity_labels(labels: List[str], xs: List[float], span: float):
    """把像素位置几乎重合的相邻标签合并为 "A|B" 断点标签（零长段）。"""
    if not labels:
        return [], []
    out_l = [labels[0]]
    out_x = [xs[0]]
    for l, x in zip(labels[1:], xs[1:]):
        # 只合并几乎重合（同一锚点）的标签对；质心间距在字符宽度量级的
        # 真断点由调用方发 warning 提示人工确认，避免双向静默误判
        if x - out_x[-1] < 0.004 * span:
            out_l[-1] = f"{out_l[-1]}|{l}"
            out_x[-1] = (out_x[-1] + x) / 2.0
        else:
            out_l.append(l)
            out_x.append(x)
    return out_l, out_x


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze one arbitrary band-structure image.")
    ap.add_argument("--image", required=True, help="能带图路径 (png/jpg)")
    ap.add_argument("--structure", default=None, help="POSCAR/CIF 结构文件（算段长用）")
    ap.add_argument("--mp", default=None, help="化学式或 mp-ID（需 MP_API_KEY）")
    ap.add_argument("--labels", default=None, help="手工给定标签序列（逗号分隔，覆盖 OCR）")
    ap.add_argument("--path_type", default="setyawan_curtarolo", choices=["setyawan_curtarolo", "hinuma", "latimer_munro"])
    ap.add_argument("--mode", default="blue_dark", choices=["blue_dark", "blue", "dark"], help="能带像素提取模式（黑色能带用 dark）")
    ap.add_argument("--dark_threshold", type=int, default=90)
    ap.add_argument("--fermi", type=float, default=0.0, help="包络划分参考能级（标定后的 eV）")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--energy_window", type=float, default=0.10)
    ap.add_argument("--local_width", type=float, default=0.10)
    ap.add_argument("--min_side_points", type=int, default=3)
    ap.add_argument("--rmse_threshold_eV", type=float, default=0.08)
    ap.add_argument("--min_quadratic_span_eV", type=float, default=0.015)
    ap.add_argument("--linear_preference_tol", type=float, default=0.10)
    ap.add_argument("--emin", type=float, default=None, help="y 轴 OCR 失败时手工给定窗口下限")
    ap.add_argument("--emax", type=float, default=None, help="y 轴 OCR 失败时手工给定窗口上限")
    ap.add_argument("--out_prefix", default=None, help="输出前缀（默认 <图名>_analysis）")
    ap.add_argument("--emit_segments_json", default=None, help="把该图的分段信息追加到指定的 kpath_segments_external.json，供 band_screen.py 批量管线使用")
    args = ap.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: cannot read image {img_path}")
        return 2
    out_prefix = Path(args.out_prefix) if args.out_prefix else img_path.with_name(img_path.stem + "_analysis")

    labels_override = [s.strip() for s in args.labels.split(",")] if args.labels else None
    axres = axes_mod.analyze_axes(str(img_path), labels_override=labels_override)
    L, R, T, B = axres.frame
    crop_box = (L + CROP_INSET, T + CROP_INSET, R - CROP_INSET, B - CROP_INSET)
    cl, ct, cr, cb = crop_box

    # ---- y 轴能量窗口 ----
    if args.emin is not None and args.emax is not None:
        emin, emax = float(args.emin), float(args.emax)
        y_source = "manual"
    elif axres.ycal is not None:
        emax = float(axres.ycal.energy_at(ct))
        emin = float(axres.ycal.energy_at(cb))
        y_source = "image_ocr"
    else:
        print("ERROR: y-axis OCR calibration failed; pass --emin/--emax manually.")
        print(f"  warnings: {axres.warnings}")
        return 3
    if not (emax > emin):
        print(f"ERROR: invalid energy window from calibration: [{emin}, {emax}]")
        return 3

    # ---- 标签 + 锚点（换算到 crop 归一化坐标）----
    labels, anchor_x = _merge_discontinuity_labels(
        axres.labels, list(axres.label_x_px), span=float(R - L)
    )
    q_anchors = [(x - cl) / max(1.0, float(cr - cl)) for x in anchor_x]
    warnings = list(axres.warnings)
    # 相邻标签间距在字符宽度量级（0.4%-2.5% 轴宽）时可能是未合并的
    # 断点对（如 X|U 以两个质心出现），提示人工用 --labels 确认
    for i in range(len(axres.labels) - 1):
        d_frac = (axres.label_x_px[i + 1] - axres.label_x_px[i]) / max(1.0, float(R - L))
        if 0.004 <= d_frac < 0.025:
            warnings.append(
                f"labels_{axres.labels[i]}_{axres.labels[i + 1]}_very_close_possible_discontinuity_not_merged"
            )
    if len(labels) < 2 or not np.all(np.diff(q_anchors) > 0):
        print(f"ERROR: label anchors invalid or non-monotonic: {list(zip(labels, q_anchors))}")
        print("  use --labels to override OCR, or check the image.")
        return 4

    # ---- 结构侧段长 ----
    seg_lengths: Optional[List[float]] = None
    seg_source = "none"
    try:
        if args.structure:
            seg_lengths = kseg.segments_from_structure(args.structure, labels, path_type=args.path_type)
            seg_source = f"structure:{args.structure}"
        elif args.mp:
            seg_lengths = kseg.segments_from_mp(args.mp, labels, path_type=args.path_type)
            seg_source = f"materials_project:{args.mp}"
    except KeyError as exc:
        print(f"ERROR: label not in this structure's standard k-path: {exc}")
        print(f"  OCR labels were: {labels}")
        print("  fix with --labels (e.g. --labels 'G,X,W,L,G') or check --path_type.")
        return 4
    except Exception as exc:
        print(f"ERROR: cannot obtain k-path segments: {exc}")
        return 4

    if seg_lengths is not None:
        if len(seg_lengths) != len(labels) - 1:
            print("ERROR: segment count != label pairs")
            return 4
        k_nodes = np.concatenate([[0.0], np.cumsum(np.asarray(seg_lengths, dtype=float))])
        kmap = bs.PiecewiseKmap(np.asarray(q_anchors, dtype=float), k_nodes, "per_segment_labels")
        k_units = "eV*A^2"
    else:
        # 无结构：无量纲恒等映射（k = 归一化 q），但保留标签锚点作为
        # 映射节点，使 v8 的顶点吸附与逐段窗口截断在无量纲模式下同样
        # 生效（否则近边界/近高对称点的拟合会退化）。
        qn = (
            np.asarray(q_anchors, dtype=float)
            if len(q_anchors) >= 2 else np.array([0.0, 1.0])
        )
        kmap = bs.PiecewiseKmap(qn, qn.copy(), "dimensionless")
        k_units = "eV (per normalized-k^2), NOT physical"
        warnings.append("no_structure_curvature_dimensionless")

    # ---- 像素提取（剔除高对称点分隔线附近的列，防实线分隔线污染包络）----
    points, _, _ = bs.extract_points(
        img, crop_box, emin=emin, emax=emax, mode=args.mode, dark_threshold=args.dark_threshold
    )
    if points.empty:
        print("ERROR: no band pixels detected in the calibrated energy window.")
        return 5
    if anchor_x:
        ax_arr = np.asarray(anchor_x, dtype=float) - cl
        px = points["x"].to_numpy(dtype=float)
        keep = np.min(np.abs(px[:, None] - ax_arr[None, :]), axis=1) > 2.0
        points = points[keep]
    q_lo_data, q_hi_data = (min(q_anchors), max(q_anchors)) if q_anchors else (0.0, 1.0)
    points = points[(points["k_norm"] >= q_lo_data - 0.005) & (points["k_norm"] <= q_hi_data + 0.005)]

    cbm_env = bs.build_envelope(points, "cbm", fermi=args.fermi, margin=args.margin)
    vbm_env = bs.build_envelope(points, "vbm", fermi=args.fermi, margin=args.margin)
    if cbm_env.empty or vbm_env.empty:
        print("ERROR: cannot build CBM/VBM envelopes (metallic image, or --fermi not inside the gap?).")
        return 6

    gap = float(cbm_env["E"].min() - vbm_env["E"].max())
    if gap <= 0.05:
        warnings.append(f"gap_le_0.05eV_possible_metal:{gap:.3f}")

    fitkw = dict(
        min_side_points=args.min_side_points,
        rmse_threshold_eV=args.rmse_threshold_eV,
        min_quadratic_span_eV=args.min_quadratic_span_eV,
        linear_preference_tol=args.linear_preference_tol,
        # 分隔线锚点列被剔除（±2px）+ 线宽，顶点偏移可达 5-7 像素，
        # 吸附容差相应放宽
        vertex_snap_cols=7.0,
    )
    cbm = bs.fit_edge_sides(cbm_env, "cbm", args.local_width, args.energy_window, kmap, **fitkw)
    vbm = bs.fit_edge_sides(vbm_env, "vbm", args.local_width, args.energy_window, kmap, **fitkw)

    def nearest_label(q: float) -> str:
        if not q_anchors:
            return ""
        i = int(np.argmin([abs(a - q) for a in q_anchors]))
        return labels[i] if abs(q_anchors[i] - q) < 0.02 else f"inside {labels[max(0, i - 1) if q < q_anchors[i] else i]}-side segment"

    # 带边位于绘图范围边缘时，真实极值顶点可能落在图外或被裁剪掉，
    # 固定顶点拟合会从假顶点出发而系统性失真。此时改用自由顶点二次
    # 拟合（E = a k² + b k + c，C = 2a）——对抛物线分支无论顶点是否
    # 可见都能无偏还原曲率。
    # 边界判据按像素自适应：crop inset(3) + mask border(4) + 线宽使
    # 最边缘数据列距锚点至少 ~7px，固定归一化阈值在窄图上会漏判。
    boundary_refit: dict = {}
    thr_boundary = max(0.012, 9.0 / float(max(cr - cl, 1)))
    col_k = kmap.total_length / float(max(cr - cl, 1))
    for name, edge in [("cbm", cbm), ("vbm", vbm)]:
        if edge is None:
            continue
        at_boundary = (
            abs(float(edge.edge_k_norm) - q_lo_data) < thr_boundary
            or abs(float(edge.edge_k_norm) - q_hi_data) < thr_boundary
        )
        if not at_boundary:
            continue
        import pandas as pd
        side_pts = pd.concat(
            [df for df in (edge.left.side_points, edge.right.side_points) if not df.empty],
            ignore_index=True,
        ) if (not edge.left.side_points.empty or not edge.right.side_points.empty) else None
        if side_pts is None or len(side_pts) < 12:
            warnings.append(f"{name}_at_plot_boundary_refit_skipped_insufficient_points")
            continue
        k_abs = kmap.k(side_pts["k_norm"].to_numpy(dtype=float))
        E_arr = side_pts["E"].to_numpy(dtype=float)
        coef = np.polyfit(k_abs, E_arr, 2)
        if abs(coef[0]) < 1e-12:
            warnings.append(f"{name}_at_plot_boundary_refit_degenerate")
            continue
        C_free = float(2.0 * coef[0])
        vertex_k = float(-coef[1] / (2 * coef[0]))
        resid = float(np.sqrt(np.mean((np.polyval(coef, k_abs) - E_arr) ** 2)))
        # 顶点合理性：自由顶点应落在带边位置附近（数个列宽内），否则
        # 属小跨度下顶点-曲率强耦合的病态解，不作为首选值。
        vertex_ok = abs(vertex_k - float(kmap.k(edge.edge_k_norm))) <= 5.0 * col_k
        preferred = "free_vertex_refit" if vertex_ok else "fixed_vertex"
        warnings.append(
            f"{name}_at_plot_boundary_"
            + ("free_vertex_refit_used" if vertex_ok else "refit_rejected_vertex_implausible")
        )
        boundary_refit[name] = {
            "preferred": preferred,
            "C_free_vertex": round(C_free, 4),
            "m_eff_m0": round(bs.HBAR2_OVER_M0_EV_A2 / abs(C_free), 4)
            if (seg_lengths is not None and abs(C_free) > 1e-9) else None,
            "vertex_k": round(vertex_k, 4),
            "vertex_units": "A^-1" if seg_lengths is not None else "normalized_q",
            "rmse_eV": round(resid, 5),
            "n_points": int(len(side_pts)),
        }

    delta_q = abs(float(cbm.edge_k_norm) - float(vbm.edge_k_norm)) if cbm and vbm else None
    delta_k = (
        abs(float(kmap.k(cbm.edge_k_norm)) - float(kmap.k(vbm.edge_k_norm)))
        if (cbm and vbm and seg_lengths is not None) else None
    )

    def side_info(sf) -> dict:
        return {
            "n_points": sf.n_points,
            "curvature": bs.opt(sf.curvature_eVA2),
            "m_eff_m0": bs.opt(sf.m_eff_m0_from_image) if seg_lengths is not None else "",
            "sign_ok": sf.sign_ok,
            "quad_rmse_eV": bs.opt(sf.quad_rmse_eV),
            "near_linear": sf.near_linear_flag,
            "reliable": sf.curvature_reliable,
        }

    result = {
        "image": str(img_path),
        "frame_LRTB": [L, R, T, B],
        "y_calibration": {
            "source": y_source,
            "emin_eV": emin,
            "emax_eV": emax,
            "n_ticks": axres.ycal.n_ticks if axres.ycal else None,
            "n_repaired": axres.ycal.n_repaired if axres.ycal else None,
            "max_resid_eV": axres.ycal.max_resid_eV if axres.ycal else None,
        },
        "labels": labels,
        "label_q_crop": [round(q, 5) for q in q_anchors],
        "anchor_sources": axres.anchor_sources,
        "segments_source": seg_source,
        "segment_lengths_Ainv": seg_lengths,
        "path_total_Ainv": float(kmap.total_length) if seg_lengths is not None else None,
        "curvature_units": k_units,
        "band_gap_eV": round(gap, 4),
        "cbm": {
            "E_eV": round(float(cbm.edge_E), 4),
            "q": round(float(cbm.edge_k_norm), 4),
            "at": nearest_label(float(cbm.edge_k_norm)),
            "left": side_info(cbm.left),
            "right": side_info(cbm.right),
        },
        "vbm": {
            "E_eV": round(float(vbm.edge_E), 4),
            "q": round(float(vbm.edge_k_norm), 4),
            "at": nearest_label(float(vbm.edge_k_norm)),
            "left": side_info(vbm.left),
            "right": side_info(vbm.right),
        },
        "direct_gap_delta_q": round(delta_q, 5) if delta_q is not None else None,
        "direct_gap_delta_k_Ainv": round(delta_k, 5) if delta_k is not None else None,
        "direct_gap_within_0p01q": bool(delta_q is not None and delta_q <= 0.01),
        "boundary_free_vertex_refit": boundary_refit,
        "warnings": warnings,
    }

    json_path = Path(str(out_prefix) + ".json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    debug_path = Path(str(out_prefix) + "_debug.png")
    title = (
        f"{img_path.stem} | gap={gap:.3f} eV | y:{y_source} [{emin:.2f},{emax:.2f}] | "
        f"calib={kmap.mode}"
    )
    bs.write_debug(img, crop_box, cbm, vbm, debug_path, emin, emax, title, separators_q=q_anchors)

    if args.emit_segments_json and seg_lengths is not None:
        ext_path = Path(args.emit_segments_json)
        data: Optional[dict] = {}
        if ext_path.exists():
            try:
                data = json.loads(ext_path.read_text(encoding="utf-8"))
            except Exception as exc:
                # 解析失败时绝不覆盖：半写/手工损坏的文件里可能有其他材料的条目
                print(f"ERROR: existing {ext_path} unreadable ({exc}); NOT overwriting it.")
                data = None
        if data is not None:
            # 键用含扩展名的文件名：band_screen.py 消费端会做 Path(k).stem，
            # 含点的文件名（Fig.3b.png）才能正确还原 material_id。
            data[img_path.name] = {
                "labels": labels,
                "boundaries_Ainv": [float(v) for v in np.concatenate([[0.0], np.cumsum(seg_lengths)])],
                "source": "external_structure_calc",
            }
            ext_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"segments appended -> {ext_path}")

    # ---- 控制台报告 ----
    print(f"== {img_path.name}")
    print(f"  y axis [{y_source}]: E in [{emin:.3f}, {emax:.3f}] eV"
          + (f" (ticks={axres.ycal.n_ticks}, repaired={axres.ycal.n_repaired}, resid={axres.ycal.max_resid_eV:.4f})" if axres.ycal else ""))
    print(f"  labels: {' - '.join(labels)}  ({seg_source})")
    if seg_lengths is not None:
        segtxt = ", ".join(f"{l1}-{l2}:{s:.4f}" for (l1, l2), s in zip(zip(labels[:-1], labels[1:]), seg_lengths))
        print(f"  segments (A^-1): {segtxt}")
    print(f"  band gap = {gap:.4f} eV | direct(|dq|<=0.01): {result['direct_gap_within_0p01q']} (dq={delta_q:.4f})")
    for name, edge in [("CBM", cbm), ("VBM", vbm)]:
        for side_name, sf in [("L", edge.left), ("R", edge.right)]:
            c = sf.curvature_eVA2
            m = sf.m_eff_m0_from_image
            ctxt = f"{c:+.3f}" if c is not None else "NA"
            mtxt = (f", m*/m0={m:.3f}" if (m is not None and seg_lengths is not None) else "")
            print(f"  {name}-{side_name}: C={ctxt} [{k_units.split(',')[0]}]{mtxt} "
                  f"(n={sf.n_points}, rmse={sf.quad_rmse_eV if sf.quad_rmse_eV is None else round(sf.quad_rmse_eV,4)}, "
                  f"near_linear={sf.near_linear_flag}, reliable={sf.curvature_reliable})")
    for name, br in boundary_refit.items():
        print(
            f"  {name.upper()} boundary free-vertex refit [{br['preferred']}]: C={br['C_free_vertex']:+.3f}"
            + (f", m*/m0={br['m_eff_m0']:.3f}" if br["m_eff_m0"] else "")
            + f" (n={br['n_points']}, rmse={br['rmse_eV']}, vertex_k={br['vertex_k']} {br['vertex_units']})"
        )
    if warnings:
        print(f"  warnings: {warnings}")
    print(f"  json  -> {json_path}")
    print(f"  debug -> {debug_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
