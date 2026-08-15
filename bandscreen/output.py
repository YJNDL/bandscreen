# -*- coding: utf-8 -*-
"""结果输出：中文列名映射与 CSV 写出。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

CN_COLUMNS = {
    "material_id": "材料名称",
    "image_file": "能带图片路径",
    "debug_image": "debug识别图片路径",
    "path_length_Ainv": "高对称路径总长度_A^-1",
    "path_length_source": "路径长度来源",
    "path_length_ocr_text": "路径长度OCR文本",
    "crop_left": "绘图区左边界_px",
    "crop_top": "绘图区上边界_px",
    "crop_right": "绘图区右边界_px",
    "crop_bottom": "绘图区下边界_px",
    "n_detected_points": "识别到的能带像素点数",
    "n_k_columns": "识别到的k列数",
    "image_gap_eV": "图像估计带隙_eV",
    "fit_energy_window_eV": "曲率拟合能量窗口_eV",
    "processing_status": "处理状态",
    "screen_out_reason": "筛除原因",
    "metallic_flag": "是否判定为金属/半金属",
    "metal_detection_reason": "金属性判定原因",
    "metal_gap_threshold_eV": "金属性小带隙阈值_eV",
    "fermi_crossing_tol_eV": "费米能级穿越容差_eV",
    "fermi_crossing_columns": "费米能级附近能带列数",
    "fermi_crossing_fraction": "费米能级附近能带列占比",
    "min_abs_band_energy_to_fermi_eV": "能带到费米能级最小距离_eV",
    "image_gap_eV_by_envelope": "包络估计带隙_eV",
    "cbm_edge_E_eV": "CBM能量_eV",
    "cbm_edge_k_norm": "CBM位置_k归一化",
    "cbm_edge_k_norm_raw": "CBM原始极值位置_k归一化",
    "cbm_vertex_snapped": "CBM顶点是否吸附到高对称点",
    "vbm_edge_E_eV": "VBM能量_eV",
    "vbm_edge_k_norm": "VBM位置_k归一化",
    "vbm_edge_k_norm_raw": "VBM原始极值位置_k归一化",
    "vbm_vertex_snapped": "VBM顶点是否吸附到高对称点",
    "manual_grade": "人工等级",
    "manual_notes": "人工备注",
    "use_for_training": "是否用于训练",
    "gap_min_eV": "带隙筛选下限_eV",
    "gap_max_eV": "带隙筛选上限_eV",
    "gap_range_selected": "是否命中0.1到0.5eV带隙筛选",
    "direct_gap_required": "是否要求直接带隙",
    "direct_gap_selected": "是否识别为直接带隙",
    "direct_gap_condition_passed": "直接带隙条件是否通过",
    "gap_and_direct_gap_selected": "是否同时满足带隙范围和直接带隙",
    "direct_gap_tolerance_k_norm": "直接带隙k点容差_归一化",
    "direct_gap_delta_k_norm": "CBM_VBM位置差_归一化k",
    "direct_gap_delta_k_Ainv": "CBM_VBM位置差_A^-1",
    "curvature_threshold_eVA2": "曲率筛选下限_eV_A2",
    "curvature_selected": "是否命中曲率筛选",
    "curvature_selected_sides": "命中曲率筛选的方向",
    "raw_curvature_threshold_sides": "原始超过曲率阈值的方向",
    "reliable_curvature_sides": "曲率可靠的方向",
    "max_abs_curvature_eVA2": "最大绝对曲率_eV_A2",
    "linear_mean_slope_threshold_eVA": "线性候选平均斜率阈值_eV_A",
    "linear_max_slope_threshold_eVA": "线性候选最大斜率阈值_eV_A",
    "linear_high_dispersion_selected": "是否命中近线性高色散筛选",
    "linear_high_dispersion_sides": "命中近线性高色散的方向",
    "near_linear_sides": "近似线性方向",
    "invalid_curvature_sign_sides": "曲率符号异常方向",
    "max_mean_abs_slope_eVA": "最大平均绝对斜率_eV_A",
    "max_max_abs_slope_eVA": "最大绝对斜率_eV_A",
    "transport_descriptor_selected": "是否命中输运描述符筛选",
    "final_selected": "是否最终筛选命中",
    "selection_reason": "筛选原因",
    "n_segments_expected": "高对称路径段数",
    "n_boundaries_detected": "图像检测到的分隔线数",
    "boundary_detect_max_err_qnorm": "分隔线检测最大偏差_归一化k",
    "k_calibration_mode": "k标定模式",
}

for _prefix_cn, _prefix_en in [
    ("CBM左侧", "cbm_left"),
    ("CBM右侧", "cbm_right"),
    ("VBM左侧", "vbm_left"),
    ("VBM右侧", "vbm_right"),
]:
    CN_COLUMNS.update({
        f"{_prefix_en}_points": f"{_prefix_cn}拟合点数",
        f"{_prefix_en}_k_span_Ainv": f"{_prefix_cn}拟合k范围_A^-1",
        f"{_prefix_en}_energy_span_eV": f"{_prefix_cn}拟合能量跨度_eV",
        f"{_prefix_en}_curvature_eVA2": f"{_prefix_cn}二阶曲率_eV_A2",
        f"{_prefix_en}_m_eff_m0_from_image": f"{_prefix_cn}图像估算有效质量_m0",
        f"{_prefix_en}_curvature_sign_ok": f"{_prefix_cn}曲率符号是否合理",
        f"{_prefix_en}_quad_rmse_eV": f"{_prefix_cn}二次拟合RMSE_eV",
        f"{_prefix_en}_linear_rmse_eV": f"{_prefix_cn}线性拟合RMSE_eV",
        f"{_prefix_en}_quadratic_energy_span_eV": f"{_prefix_cn}二次项贡献能量_eV",
        f"{_prefix_en}_near_linear_flag": f"{_prefix_cn}是否近似线性",
        f"{_prefix_en}_curvature_reliable": f"{_prefix_cn}曲率是否可靠",
        f"{_prefix_en}_linear_slope_eVA": f"{_prefix_cn}线性斜率_eV_A",
        f"{_prefix_en}_mean_abs_slope_eVA": f"{_prefix_cn}平均绝对斜率_eV_A",
        f"{_prefix_en}_max_abs_slope_eVA": f"{_prefix_cn}最大绝对斜率_eV_A",
    })


def to_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: CN_COLUMNS.get(c, c) for c in df.columns})


RESULT_CURVATURE_COLS = [
    "cbm_left_curvature_eVA2",
    "cbm_right_curvature_eVA2",
    "vbm_left_curvature_eVA2",
    "vbm_right_curvature_eVA2",
]

COMPACT_CN_RENAME = {
    "cbm_left_curvature_eVA2": "CBM左侧绝对曲率_eV_A2",
    "cbm_right_curvature_eVA2": "CBM右侧绝对曲率_eV_A2",
    "vbm_left_curvature_eVA2": "VBM左侧绝对曲率_eV_A2",
    "vbm_right_curvature_eVA2": "VBM右侧绝对曲率_eV_A2",
}


def write_result_csvs(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """写出四个结果 CSV：精简曲率表（中/英）+ 全量明细（中/英）。

    精简表只含四个方向的绝对曲率值，material_id 作为行索引。
    """
    full_en = output_dir / "band_curvature_slope_results_english.csv"
    full_cn = output_dir / "band_curvature_slope_results.csv"

    compact_results = pd.DataFrame(index=df.index)
    for c in RESULT_CURVATURE_COLS:
        if c in df.columns:
            compact_results[c] = pd.to_numeric(df[c], errors="coerce").abs()
        else:
            compact_results[c] = np.nan

    if "material_id" in df.columns:
        compact_results.index = df["material_id"].astype(str)
    else:
        compact_results.index = [str(i) for i in range(len(compact_results))]
    compact_results.index.name = "material_id"
    compact_results.to_csv(full_en, index=True, encoding="utf-8-sig")

    compact_cn = compact_results.rename(columns=COMPACT_CN_RENAME)
    compact_cn.index.name = "材料名称"
    compact_cn.to_csv(full_cn, index=True, encoding="utf-8-sig")

    # 全量明细：k_calibration_mode、分隔线检测偏差等诊断列都在其中。
    detail_en = output_dir / "band_curvature_full_details_english.csv"
    detail_cn = output_dir / "band_curvature_full_details.csv"
    df.to_csv(detail_en, index=False, encoding="utf-8-sig")
    to_chinese_columns(df).to_csv(detail_cn, index=False, encoding="utf-8-sig")

    return {"compact_en": full_en, "compact_cn": full_cn, "detail_en": detail_en, "detail_cn": detail_cn}
