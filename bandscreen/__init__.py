# -*- coding: utf-8 -*-
"""bandscreen —— 能带图像筛选工具包。

模块一览：
    common     常量与小工具（safe_stem / opt / HBAR2_OVER_M0_EV_A2 ...）
    kpath      晶体结构 -> 高对称路径段长（Å^-1），任意图模式的结构侧输入
    render     DAT/HDF5 -> 标准化 PNG + kpath_segments.json
    extract    绘图区检测、能带像素提取、CBM/VBM 包络
    calibrate  q<->k 分段线性标定（PiecewiseKmap）+ 分隔线检测 + OCR 回退
    fitting    固定顶点二次拟合（顶点吸附 / E0 去偏 / 段边界截断）
    draw       debug 叠加图
    screening  金属判定、单图处理 process_one、候选评估、直接带隙
    output     中文列名映射与 CSV 写出
    axes_ocr   任意文献图的坐标轴识别（y 轴刻度 OCR、高对称标签、锚点）

入口脚本（包外同目录）：
    band_screen.py         批量筛选 CLI（交互式，替代 v7/v8 单文件脚本）
    analyze_band_image.py  任意单图分析 CLI
"""

from .calibrate import PiecewiseKmap, build_kmap, detect_vertical_separators, ocr_path_length_from_image
from .common import HBAR2_OVER_M0_EV_A2, IMAGE_EXTS, guess_material_id, list_images, opt, safe_stem
from .draw import write_debug, write_metal_debug
from .extract import auto_detect_plot_box, build_envelope, extract_points, make_band_mask, parse_crop
from .fitting import EdgeResult, SideFit, fit_edge_sides
from .output import CN_COLUMNS, to_chinese_columns, write_result_csvs
from .render import prepare_standard_band_images
from .screening import (
    apply_direct_gap_screening,
    copy_selected_debug_images,
    detect_metallicity,
    evaluate_selection,
    process_one,
    to_bool_series,
)

__all__ = [
    "PiecewiseKmap",
    "build_kmap",
    "detect_vertical_separators",
    "ocr_path_length_from_image",
    "HBAR2_OVER_M0_EV_A2",
    "IMAGE_EXTS",
    "guess_material_id",
    "list_images",
    "opt",
    "safe_stem",
    "write_debug",
    "write_metal_debug",
    "auto_detect_plot_box",
    "build_envelope",
    "extract_points",
    "make_band_mask",
    "parse_crop",
    "EdgeResult",
    "SideFit",
    "fit_edge_sides",
    "CN_COLUMNS",
    "to_chinese_columns",
    "write_result_csvs",
    "prepare_standard_band_images",
    "apply_direct_gap_screening",
    "copy_selected_debug_images",
    "detect_metallicity",
    "evaluate_selection",
    "process_one",
    "to_bool_series",
]
