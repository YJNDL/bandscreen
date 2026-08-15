# -*- coding: utf-8 -*-
"""单图处理与材料级筛选：金属判定、process_one、候选评估、直接带隙。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .calibrate import build_kmap, detect_vertical_separators, ocr_path_length_from_image
from .common import finite_float, guess_material_id, opt
from .draw import write_debug, write_metal_debug
from .extract import build_envelope, extract_points, parse_crop
from .fitting import EdgeResult, SideFit, _empty_side, fit_edge_sides


def detect_metallicity(
    points: pd.DataFrame,
    cbm_env: pd.DataFrame,
    vbm_env: pd.DataFrame,
    fermi: float,
    fermi_crossing_tol_eV: float,
    metal_gap_threshold_eV: float,
    min_fermi_crossing_columns: int,
    min_fermi_crossing_fraction: float,
) -> Dict[str, object]:
    """Detect metallic or semimetallic band images before CBM/VBM fitting."""
    n_cols = int(points["x"].nunique()) if points is not None and not points.empty and "x" in points.columns else 0
    min_abs_to_fermi = None
    fermi_crossing_columns = 0
    fermi_crossing_fraction = 0.0
    if points is not None and not points.empty:
        abs_to_fermi = np.abs(points["E"].to_numpy(dtype=float) - float(fermi))
        if abs_to_fermi.size:
            min_abs_to_fermi = float(np.min(abs_to_fermi))
        near_fermi = points[np.abs(points["E"] - fermi) <= fermi_crossing_tol_eV]
        # 剔除近水平细线（如文献图中深色的 E=0 虚线）：同一像素行覆盖
        # 大量 k 列的近费米点不计入金属性穿越统计。
        if not near_fermi.empty and n_cols > 0:
            y_round = near_fermi["y"].round(0)
            row_counts = near_fermi.groupby(y_round)["x"].nunique()
            wide_rows = set(row_counts[row_counts > 0.25 * n_cols].index)
            if wide_rows:
                near_fermi = near_fermi[~y_round.isin(wide_rows)]
        if not near_fermi.empty:
            fermi_crossing_columns = int(near_fermi["x"].nunique())
        if n_cols > 0:
            fermi_crossing_fraction = float(fermi_crossing_columns / n_cols)

    image_gap = None
    if cbm_env is not None and vbm_env is not None and (not cbm_env.empty) and (not vbm_env.empty):
        image_gap = float(cbm_env["E"].min() - vbm_env["E"].max())

    gap_metal_hit = bool(image_gap is not None and np.isfinite(image_gap) and image_gap <= metal_gap_threshold_eV)
    fermi_pixel_hit = bool(
        fermi_crossing_columns >= min_fermi_crossing_columns
        or fermi_crossing_fraction >= min_fermi_crossing_fraction
    )
    metallic_flag = bool(gap_metal_hit or fermi_pixel_hit)
    reasons: List[str] = []
    if gap_metal_hit:
        reasons.append(f"image_gap<={metal_gap_threshold_eV:.3f}eV")
    if fermi_pixel_hit:
        reasons.append(f"fermi_pixels:{fermi_crossing_columns}cols/{fermi_crossing_fraction:.4f}")
    if not reasons:
        reasons.append("not_metallic")
    return {
        "metallic_flag": metallic_flag,
        "metal_detection_reason": ";".join(reasons),
        "metal_gap_threshold_eV": metal_gap_threshold_eV,
        "fermi_crossing_tol_eV": fermi_crossing_tol_eV,
        "fermi_crossing_columns": fermi_crossing_columns,
        "fermi_crossing_fraction": fermi_crossing_fraction,
        "min_abs_band_energy_to_fermi_eV": opt(min_abs_to_fermi),
        "image_gap_eV_by_envelope": opt(image_gap),
    }


def _side_label(kind: str, side: str) -> str:
    return f"{kind.upper()}-{side}"


def _collect_side_fits(cbm: Optional[EdgeResult], vbm: Optional[EdgeResult]) -> List[Tuple[str, SideFit]]:
    """Collect all four side fits with labels."""
    pairs: List[Tuple[str, SideFit]] = []
    if cbm is not None:
        pairs.append((_side_label("cbm", "left"), cbm.left))
        pairs.append((_side_label("cbm", "right"), cbm.right))
    if vbm is not None:
        pairs.append((_side_label("vbm", "left"), vbm.left))
        pairs.append((_side_label("vbm", "right"), vbm.right))
    return pairs


def evaluate_selection(
    cbm: Optional[EdgeResult],
    vbm: Optional[EdgeResult],
    image_gap_eV: Optional[float],
    gap_min_eV: float,
    gap_max_eV: float,
    curvature_min_eVA2: float,
    curvature_max_eVA2: float,
    linear_mean_slope_threshold_eVA: float,
    linear_max_slope_threshold_eVA: float,
) -> Dict[str, object]:
    """
    材料级筛选标记。

    带隙：图像估计带隙落在 [gap_min_eV, gap_max_eV] 内；
    曲率：任一 CBM/VBM 方向符号正确且 |C| 落在给定范围内；
    近线性高色散：near_linear 分支按斜率阈值命中；
    最终候选 = 带隙命中 AND（曲率命中 OR 近线性高色散命中），
    直接带隙条件（如启用）由 apply_direct_gap_screening 事后应用。
    """
    side_fits = _collect_side_fits(cbm, vbm)

    gap_val = finite_float(image_gap_eV)
    gap_selected = bool(
        gap_val is not None and gap_min_eV <= gap_val <= gap_max_eV
    )

    curvature_sides: List[str] = []
    raw_curvature_sides: List[str] = []
    linear_sides: List[str] = []
    reliable_curvature_sides: List[str] = []
    near_linear_sides: List[str] = []
    invalid_sign_sides: List[str] = []

    max_abs_curvature = None
    max_mean_abs_slope = None
    max_max_abs_slope = None

    for label, sf in side_fits:
        C = finite_float(sf.curvature_eVA2)
        mean_slope = finite_float(sf.mean_abs_slope_eVA)
        max_slope = finite_float(sf.max_abs_slope_eVA)

        if C is not None:
            absC = abs(C)
            max_abs_curvature = (
                absC if max_abs_curvature is None
                else max(max_abs_curvature, absC)
            )

            in_curvature_range = bool(
                curvature_min_eVA2 <= absC <= curvature_max_eVA2
            )
            if in_curvature_range:
                # Keep the legacy column name raw_curvature_threshold_sides
                # so downstream output schemas/files do not change.
                raw_curvature_sides.append(label)
                if sf.sign_ok:
                    curvature_sides.append(label)

        if sf.curvature_reliable:
            reliable_curvature_sides.append(label)
        if sf.near_linear_flag:
            near_linear_sides.append(label)
        if C is not None and not sf.sign_ok:
            invalid_sign_sides.append(label)

        if mean_slope is not None:
            max_mean_abs_slope = (
                mean_slope if max_mean_abs_slope is None
                else max(max_mean_abs_slope, mean_slope)
            )
        if max_slope is not None:
            max_max_abs_slope = (
                max_slope if max_max_abs_slope is None
                else max(max_max_abs_slope, max_slope)
            )

        if sf.near_linear_flag:
            mean_hit = (
                mean_slope is not None
                and mean_slope >= linear_mean_slope_threshold_eVA
            )
            max_hit = (
                max_slope is not None
                and max_slope >= linear_max_slope_threshold_eVA
            )
            if mean_hit or max_hit:
                linear_sides.append(label)

    curvature_selected = len(curvature_sides) > 0
    linear_selected = len(linear_sides) > 0
    transport_selected = curvature_selected or linear_selected
    final_selected = gap_selected and transport_selected

    max_txt = (
        "inf" if not np.isfinite(curvature_max_eVA2)
        else f"{curvature_max_eVA2:.3f}"
    )

    reasons: List[str] = []
    if gap_selected:
        reasons.append(
            f"gap_hit:{gap_min_eV:.3f}-{gap_max_eV:.3f}eV"
        )
    else:
        reasons.append("gap_not_in_range")

    if curvature_selected:
        reasons.append(
            f"curvature_range_hit:{curvature_min_eVA2:.3f}-{max_txt}eVA2:"
            f"{';'.join(curvature_sides)}"
        )
    if linear_selected:
        reasons.append(
            f"linear_high_dispersion:{';'.join(linear_sides)}"
        )
    if not transport_selected:
        reasons.append("transport_descriptor_not_selected")

    return {
        "gap_min_eV": gap_min_eV,
        "gap_max_eV": gap_max_eV,
        "gap_range_selected": gap_selected,

        # Backward-compatible output column:
        # it now stores the lower bound of the requested |curvature| range.
        "curvature_threshold_eVA2": curvature_min_eVA2,

        "curvature_selected": curvature_selected,
        "curvature_selected_sides": ";".join(curvature_sides),
        "raw_curvature_threshold_sides": ";".join(raw_curvature_sides),
        "reliable_curvature_sides": ";".join(reliable_curvature_sides),
        "max_abs_curvature_eVA2": opt(max_abs_curvature),

        "linear_mean_slope_threshold_eVA": linear_mean_slope_threshold_eVA,
        "linear_max_slope_threshold_eVA": linear_max_slope_threshold_eVA,
        "linear_high_dispersion_selected": linear_selected,
        "linear_high_dispersion_sides": ";".join(linear_sides),
        "near_linear_sides": ";".join(near_linear_sides),
        "invalid_curvature_sign_sides": ";".join(invalid_sign_sides),
        "max_mean_abs_slope_eVA": opt(max_mean_abs_slope),
        "max_max_abs_slope_eVA": opt(max_max_abs_slope),
        "transport_descriptor_selected": transport_selected,
        "final_selected": final_selected,
        "selection_reason": "; ".join(reasons),
    }


def copy_selected_debug_images(df: pd.DataFrame, output_dir: Path) -> Dict[str, int]:
    """Copy debug images for selected materials into separated folders."""
    selected_root = output_dir / "selected_debug_images"
    folders = {
        "final_selected": selected_root / "final_selected",
        "gap_range_selected": selected_root / "gap_0p1_0p5_semiconductor",
        "curvature_selected": selected_root / "curvature_selected_absC_ge_threshold",
        "linear_high_dispersion_selected": selected_root / "linear_high_dispersion",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    counts = {k: 0 for k in folders}
    if df.empty or "debug_image" not in df.columns:
        return counts

    for _, row in df.iterrows():
        debug = str(row.get("debug_image", "") or "")
        if not debug:
            continue
        src = Path(debug)
        if not src.exists():
            continue
        for key, folder in folders.items():
            if bool(row.get(key, False)):
                dst = folder / src.name
                try:
                    shutil.copy2(src, dst)
                    counts[key] += 1
                except Exception:
                    pass
    return counts


def _base_screenout_row(material_id: str, image_path: Path, status: str, reason: str) -> Dict[str, object]:
    """Create a non-error row for images that should be screened out gracefully."""
    return {
        "material_id": material_id,
        "image_file": str(image_path),
        "debug_image": "",
        "path_length_Ainv": "",
        "path_length_source": status,
        "path_length_ocr_text": "",
        "crop_left": "",
        "crop_top": "",
        "crop_right": "",
        "crop_bottom": "",
        "n_detected_points": 0,
        "n_k_columns": 0,
        "image_gap_eV": "",
        "metallic_flag": False,
        "metal_detection_reason": "",
        "metal_gap_threshold_eV": "",
        "fermi_crossing_tol_eV": "",
        "fermi_crossing_columns": "",
        "fermi_crossing_fraction": "",
        "min_abs_band_energy_to_fermi_eV": "",
        "image_gap_eV_by_envelope": "",
        "processing_status": status,
        "screen_out_reason": reason,
        "n_segments_expected": "",
        "n_boundaries_detected": "",
        "boundary_detect_max_err_qnorm": "",
        "k_calibration_mode": "not_attempted",
        "gap_min_eV": "",
        "gap_max_eV": "",
        "gap_range_selected": False,
        "curvature_selected": False,
        "linear_high_dispersion_selected": False,
        "transport_descriptor_selected": False,
        "final_selected": False,
        "selection_reason": reason,
        "manual_grade": "",
        "manual_notes": reason,
        "use_for_training": 0,
    }


def process_one(
    image_path: Path,
    args: argparse.Namespace,
    segments_map: Dict[str, dict],
    output_debug_dir: Path,
    ocr_region_dir: Optional[Path],
) -> Dict[str, object]:
    """
    处理一张标准化能带图。

    像素->Å^-1 映射按高对称路径逐段标定：结构侧段边界（segments_map）
    + 图像检测的灰色分隔线位置。无分段信息的图像回退到 OCR 读取图上
    印的 Path length 总长做全局均匀标定。

    流程约定：
    - 能量窗口内无能带像素、或无法构造 CBM/VBM 包络：正常筛除；
    - 金属/半金属（费米穿越或带隙过小）：筛除并出专用 debug 图；
    - 只要带边和 k 标定可用，就总是输出四个方向的曲率/斜率描述符；
      带隙条件只用于 final_selected，不阻断曲率输出。
    """
    material_id = guess_material_id(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        return _base_screenout_row(material_id, image_path, "screened_out", "cannot_read_image")

    crop_box = parse_crop(args.crop, img)
    points, _, _ = extract_points(
        img,
        crop_box,
        emin=args.emin,
        emax=args.emax,
        mode=args.mode,
        dark_threshold=args.dark_threshold,
    )

    if points.empty:
        row = _base_screenout_row(
            material_id,
            image_path,
            "screened_out",
            "no_band_pixels_in_energy_window_or_gap_larger_than_window",
        )
        row.update({
            "crop_left": crop_box[0],
            "crop_top": crop_box[1],
            "crop_right": crop_box[2],
            "crop_bottom": crop_box[3],
        })
        return row

    cbm_env = build_envelope(points, "cbm", fermi=args.fermi, margin=args.margin)
    vbm_env = build_envelope(points, "vbm", fermi=args.fermi, margin=args.margin)

    if cbm_env.empty or vbm_env.empty:
        row = _base_screenout_row(
            material_id,
            image_path,
            "screened_out",
            "missing_cbm_or_vbm_envelope_in_energy_window",
        )
        row.update({
            "crop_left": crop_box[0],
            "crop_top": crop_box[1],
            "crop_right": crop_box[2],
            "crop_bottom": crop_box[3],
            "n_detected_points": int(len(points)),
            "n_k_columns": int(points["x"].nunique()),
        })
        return row

    cbm_E_tmp = float(cbm_env["E"].min())
    vbm_E_tmp = float(vbm_env["E"].max())
    gap_tmp = cbm_E_tmp - vbm_E_tmp
    gap_range_selected_tmp = bool(args.gap_min_eV <= gap_tmp <= args.gap_max_eV)

    # Metal/semimetal detection before any CBM/VBM curvature fitting.
    metal_info = detect_metallicity(
        points=points,
        cbm_env=cbm_env,
        vbm_env=vbm_env,
        fermi=args.fermi,
        fermi_crossing_tol_eV=args.fermi_crossing_tol_eV,
        metal_gap_threshold_eV=args.metal_gap_threshold_eV,
        min_fermi_crossing_columns=args.min_fermi_crossing_columns,
        min_fermi_crossing_fraction=args.min_fermi_crossing_fraction,
    )
    if bool(metal_info.get("metallic_flag", False)):
        debug_path_obj = output_debug_dir / f"{material_id}_metal_debug.png"
        debug_path = ""
        if not args.no_debug:
            title = f"{material_id} | METAL/SEMIMETAL | {metal_info.get('metal_detection_reason', '')}"
            write_metal_debug(
                img, crop_box, points, debug_path_obj,
                args.emin, args.emax, args.fermi, args.fermi_crossing_tol_eV, title
            )
            debug_path = str(debug_path_obj)
        row = _base_screenout_row(
            material_id,
            image_path,
            "screened_out",
            "metal_or_semimetal_fermi_crossing_no_cbm_vbm_fit",
        )
        row.update({
            "debug_image": debug_path,
            "crop_left": crop_box[0],
            "crop_top": crop_box[1],
            "crop_right": crop_box[2],
            "crop_bottom": crop_box[3],
            "n_detected_points": int(len(points)),
            "n_k_columns": int(points["x"].nunique()),
            "image_gap_eV": gap_tmp,
            "cbm_edge_E_eV": cbm_E_tmp,
            "vbm_edge_E_eV": vbm_E_tmp,
            "selection_reason": "metallic_screened_out_no_cbm_vbm_fit",
        })
        row.update(metal_info)
        return row

    # k 标定：优先使用结构侧分段元数据 + 图像检测到的分隔线；无分段
    # 信息的图像回退到 OCR 读取图上印的 Path length 总长。
    seg_info = segments_map.get(material_id) if segments_map else None
    k_total: Optional[float] = None
    path_src, path_raw = "", ""
    if seg_info is not None:
        b_arr = np.asarray(seg_info.get("boundaries_Ainv", []), dtype=float)
        if b_arr.size >= 2 and np.all(np.diff(b_arr) >= -1e-12) and (b_arr[-1] - b_arr[0]) > 1e-9:
            k_total = float(b_arr[-1] - b_arr[0])
            path_src = "structure_segments"
        else:
            seg_info = None
    if seg_info is None:
        ocr_region_out = (ocr_region_dir / f"{material_id}_ocr_region.png") if ocr_region_dir is not None else None
        k_total, path_src, path_raw = ocr_path_length_from_image(
            img, tesseract_cmd=args.tesseract_cmd, ocr_region_out=ocr_region_out
        )

    cl, ct, cr, cb = crop_box
    detected_q = detect_vertical_separators(img[ct:cb + 1, cl:cr + 1])
    # auto 裁剪时绘图框 spine 中心位于 crop 边缘外约 inset(=3) 像素处，
    # 端点节点据此外扩，避免端点偏差集中到首末两段。
    crop_is_auto = args.crop is None or str(args.crop).strip().lower() == "auto"
    q_pad = (3.0 / max(cr - cl, 1)) if crop_is_auto else 0.0
    kmap, kdiag = build_kmap(seg_info, detected_q, k_total, q_pad=q_pad)
    if kmap is None:
        row = _base_screenout_row(
            material_id,
            image_path,
            "screened_out",
            "path_length_unreadable_after_band_detected",
        )
        row.update(kdiag)
        row.update({
            "path_length_source": path_src,
            "path_length_ocr_text": path_raw,
            "crop_left": crop_box[0],
            "crop_top": crop_box[1],
            "crop_right": crop_box[2],
            "crop_bottom": crop_box[3],
            "n_detected_points": int(len(points)),
            "n_k_columns": int(points["x"].nunique()),
            "image_gap_eV": gap_tmp,
            "gap_min_eV": args.gap_min_eV,
            "gap_max_eV": args.gap_max_eV,
            "gap_range_selected": gap_range_selected_tmp,
            "fit_energy_window_eV": args.energy_window,
        })
        row.update(metal_info)
        return row

    fitkw = dict(
        min_side_points=args.min_side_points,
        rmse_threshold_eV=args.rmse_threshold_eV,
        min_quadratic_span_eV=args.min_quadratic_span_eV,
        linear_preference_tol=args.linear_preference_tol,
    )
    cbm = fit_edge_sides(cbm_env, "cbm", args.local_width, args.energy_window, kmap, **fitkw)
    vbm = fit_edge_sides(vbm_env, "vbm", args.local_width, args.energy_window, kmap, **fitkw)

    cbm_E = cbm.edge_E if cbm is not None else np.nan
    vbm_E = vbm.edge_E if vbm is not None else np.nan
    gap = float(cbm_E - vbm_E) if np.isfinite(cbm_E) and np.isfinite(vbm_E) else gap_tmp

    debug_path_obj = output_debug_dir / f"{material_id}_debug.png"
    if not args.no_debug:
        def mtxt(side: Optional[SideFit]) -> str:
            if side is None or side.m_eff_m0_from_image is None:
                return "NA"
            return f"{side.m_eff_m0_from_image:.2f}"
        title = (
            f"{material_id} | gap={gap:.3f} eV | L={kmap.total_length:.4f} A^-1 | "
            f"calib={kmap.mode} | "
            f"m*: CBM L/R={mtxt(cbm.left if cbm else None)}/{mtxt(cbm.right if cbm else None)}, "
            f"VBM L/R={mtxt(vbm.left if vbm else None)}/{mtxt(vbm.right if vbm else None)}"
        )
        write_debug(img, crop_box, cbm, vbm, debug_path_obj, args.emin, args.emax, title,
                    separators_q=detected_q)
    debug_path = str(debug_path_obj) if not args.no_debug else ""

    row: Dict[str, object] = {
        "material_id": material_id,
        "image_file": str(image_path),
        "debug_image": debug_path,
        "path_length_Ainv": opt(k_total),
        "path_length_source": path_src,
        "path_length_ocr_text": path_raw,
        "crop_left": crop_box[0],
        "crop_top": crop_box[1],
        "crop_right": crop_box[2],
        "crop_bottom": crop_box[3],
        "n_detected_points": int(len(points)),
        "n_k_columns": int(points["x"].nunique()),
        "image_gap_eV": gap,
        "fit_energy_window_eV": args.energy_window,
        "processing_status": "ok",
        # 与 evaluate_selection 同口径：用 E0 重估后的 gap 判断，
        # 避免 reason 列与筛选列自相矛盾（包络原始值另存
        # image_gap_eV_by_envelope 列）。
        "screen_out_reason": "" if (args.gap_min_eV <= gap <= args.gap_max_eV) else "gap_out_of_range_but_curvature_computed",
        "cbm_edge_E_eV": opt(cbm.edge_E if cbm else None),
        "cbm_edge_k_norm": opt(cbm.edge_k_norm if cbm else None),
        "cbm_edge_k_norm_raw": opt(cbm.edge_k_norm_raw if cbm else None),
        "cbm_vertex_snapped": bool(cbm.vertex_snapped) if cbm else False,
        "vbm_edge_E_eV": opt(vbm.edge_E if vbm else None),
        "vbm_edge_k_norm": opt(vbm.edge_k_norm if vbm else None),
        "vbm_edge_k_norm_raw": opt(vbm.edge_k_norm_raw if vbm else None),
        "vbm_vertex_snapped": bool(vbm.vertex_snapped) if vbm else False,
        "manual_grade": "",
        "manual_notes": "",
        "use_for_training": "",
    }
    row.update(metal_info)
    row.update(kdiag)
    # CBM-VBM 的 Å^-1 距离按分段映射计算（apply_direct_gap_screening
    # 仅在该值缺失时用"归一化差×总长"的均匀近似回退）。
    if cbm is not None and vbm is not None:
        row["direct_gap_delta_k_Ainv"] = opt(
            abs(float(kmap.k(cbm.edge_k_norm)) - float(kmap.k(vbm.edge_k_norm)))
        )

    def add_side(prefix: str, side_fit: Optional[SideFit]):
        if side_fit is None:
            side_fit = _empty_side(prefix)
        row.update({
            f"{prefix}_points": side_fit.n_points,
            f"{prefix}_k_span_Ainv": opt(side_fit.k_span_Ainv),
            f"{prefix}_energy_span_eV": opt(side_fit.energy_span_eV),
            f"{prefix}_curvature_eVA2": opt(side_fit.curvature_eVA2),
            f"{prefix}_m_eff_m0_from_image": opt(side_fit.m_eff_m0_from_image),
            f"{prefix}_curvature_sign_ok": side_fit.sign_ok,
            f"{prefix}_quad_rmse_eV": opt(side_fit.quad_rmse_eV),
            f"{prefix}_linear_rmse_eV": opt(side_fit.linear_rmse_eV),
            f"{prefix}_quadratic_energy_span_eV": opt(side_fit.quadratic_energy_span_eV),
            f"{prefix}_near_linear_flag": side_fit.near_linear_flag,
            f"{prefix}_curvature_reliable": side_fit.curvature_reliable,
            f"{prefix}_linear_slope_eVA": opt(side_fit.linear_slope_eVA),
            f"{prefix}_mean_abs_slope_eVA": opt(side_fit.mean_abs_slope_eVA),
            f"{prefix}_max_abs_slope_eVA": opt(side_fit.max_abs_slope_eVA),
        })

    add_side("cbm_left", cbm.left if cbm else None)
    add_side("cbm_right", cbm.right if cbm else None)
    add_side("vbm_left", vbm.left if vbm else None)
    add_side("vbm_right", vbm.right if vbm else None)

    # Gap is now used only for final selection; curvature columns are always output
    # when a usable band edge and path length exist.
    row.update(evaluate_selection(
        cbm=cbm,
        vbm=vbm,
        image_gap_eV=gap,
        gap_min_eV=args.gap_min_eV,
        gap_max_eV=args.gap_max_eV,
        curvature_min_eVA2=args.curvature_min_eVA2,
        curvature_max_eVA2=args.curvature_max_eVA2,
        linear_mean_slope_threshold_eVA=args.linear_mean_slope_threshold_eVA,
        linear_max_slope_threshold_eVA=args.linear_max_slope_threshold_eVA,
    ))

    return row


def to_bool_series(s: pd.Series) -> pd.Series:
    """Convert mixed bool/string/numeric pandas Series to bool."""
    return s.map(lambda x: bool(x) if isinstance(x, (bool, np.bool_)) else str(x).strip().lower() in ["true", "1", "yes", "y"])


def apply_direct_gap_screening(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Add direct-band-gap descriptors and optionally require direct gap for final selection.

    Direct gap is judged from the image-detected CBM/VBM k positions:
        |k_CBM - k_VBM| <= direct_gap_tolerance_norm
    where k is the normalized high-symmetry path coordinate.
    """
    if df is None or df.empty:
        return df

    cbm_k = pd.to_numeric(df.get("cbm_edge_k_norm", pd.Series(index=df.index, dtype=float)), errors="coerce")
    vbm_k = pd.to_numeric(df.get("vbm_edge_k_norm", pd.Series(index=df.index, dtype=float)), errors="coerce")
    delta_k_norm = (cbm_k - vbm_k).abs()

    direct_selected = delta_k_norm.notna() & (delta_k_norm <= float(args.direct_gap_tolerance_norm))

    df["direct_gap_required"] = bool(args.direct_gap_only)
    df["direct_gap_tolerance_k_norm"] = float(args.direct_gap_tolerance_norm)
    df["direct_gap_delta_k_norm"] = delta_k_norm
    df["direct_gap_selected"] = direct_selected

    # 优先使用 process_one 里按分段映射算出的 Å^-1 距离，
    # 缺失时才退回"归一化差×总长"的全局均匀近似。
    precomputed = pd.to_numeric(
        df.get("direct_gap_delta_k_Ainv", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    )
    if "path_length_Ainv" in df.columns:
        path_len = pd.to_numeric(df["path_length_Ainv"], errors="coerce")
        fallback = delta_k_norm * path_len
    else:
        fallback = pd.Series(np.nan, index=df.index)
    df["direct_gap_delta_k_Ainv"] = precomputed.where(precomputed.notna(), fallback)

    # Explicit combined descriptor: material simultaneously satisfies
    # the requested band-gap range and the direct-gap criterion.
    if "gap_range_selected" in df.columns:
        gap_selected = to_bool_series(df["gap_range_selected"])
    else:
        gap_selected = pd.Series(False, index=df.index)
    df["gap_and_direct_gap_selected"] = gap_selected & direct_selected

    df["direct_gap_condition_passed"] = True
    if bool(args.direct_gap_only):
        df["direct_gap_condition_passed"] = direct_selected
        if "final_selected" in df.columns:
            old_final = to_bool_series(df["final_selected"])
        else:
            old_final = pd.Series(False, index=df.index)
        df["final_selected"] = old_final & direct_selected

        if "selection_reason" not in df.columns:
            df["selection_reason"] = ""
        reason = df["selection_reason"].fillna("").astype(str)
        df["selection_reason"] = reason + np.where(
            direct_selected,
            "; direct_gap_hit",
            "; not_direct_gap",
        )

    return df
