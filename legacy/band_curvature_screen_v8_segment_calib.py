#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
band_curvature_screen_v8_segment_calib.py

v8 逐段标定版：在 v7 基础上，把横轴像素→物理波矢（Å^-1）的换算从
"全局 Path length 总长"升级为"高对称路径逐段标定"：

    1. 渲染阶段（DAT/HDF5 -> 标准 PNG）额外输出每个材料的高对称点
       标签序列与各段边界的累积 k 坐标（kpath_segments.json，属于
       声明的晶体侧输入，不含任何绘图像素信息）；
    2. 筛选阶段从图像像素检测高对称点处的灰色竖直分隔线位置，与
       结构侧段长逐段配对，得到分段线性 q->k 映射（PiecewiseKmap）；
    3. 带边曲率拟合只使用带边所在段的局部标定，天然兼容"各段像素
       宽度不按物理长度等比绘制"的图（文献图常见画法）；
    4. 无分段信息的图像回退到 v7 的全局标定链（外部 CSV / OCR /
       --ktotal），输出列 k_calibration_mode 标明每行走的是哪条路。

其余流程与 v7 相同：交互式设置带隙范围、直接带隙、拟合能量窗口和
|C| 筛选范围；每轮清空输出目录；输出精简曲率 CSV（中英文），v8 另
增全量明细 CSV 便于审计逐段标定的质量指标。

默认目录结构：
    当前目录/
    ├── band_curvature_linear_screen_v4.py
    └── band_images/
        ├── Ba4-Mg4-Si4.png
        ├── Sn4-Te4.png
        └── ...

直接运行：
    python band_curvature_linear_screen_v4.py

Windows 如需显式指定 OCR：
    python band_curvature_linear_screen_v4.py --tesseract_cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

说明：
    本脚本基于能带图片像素识别，输出的是用于高通量预筛的图像近似描述符，
    不替代基于原始能带数据的严格有效质量/迁移率计算。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
HBAR2_OVER_M0_EV_A2 = 7.619964  # ħ²/m0, unit: eV·Å²


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


def safe_stem(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", str(name).strip())
    name = re.sub(r"\s+", "-", name)
    return name or "unknown"


def list_images(input_dir: Path) -> List[Path]:
    images: List[Path] = []
    for ext in IMAGE_EXTS:
        images.extend(input_dir.glob(ext))
    return sorted(images)


def guess_material_id(path: Path) -> str:
    return safe_stem(path.stem)


def load_path_lengths(csv_path: Optional[Path]) -> Dict[str, float]:
    if csv_path is None or not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    key_col = None
    for c in ["material_id", "sample_id", "file", "filename", "image", "name", "材料名称"]:
        if c in df.columns:
            key_col = c
            break
    val_col = None
    for c in ["path_length_Ainv", "ktotal", "k_total", "path_length", "Path length", "路径长度_A^-1", "高对称路径总长度_A^-1"]:
        if c in df.columns:
            val_col = c
            break
    if key_col is None or val_col is None:
        raise ValueError(f"Cannot parse {csv_path}. Need columns like material_id,path_length_Ainv")
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        key = safe_stem(Path(str(row[key_col])).stem)
        try:
            val = float(row[val_col])
        except Exception:
            continue
        if np.isfinite(val) and val > 0:
            out[key] = val
    return out


def parse_path_length_from_text(text: str) -> Optional[float]:
    """
    Robustly parse the numeric value after "Path length" from OCR text.

    OCR may read "Path length: 12.8166 Å^-1" as variants such as:
    "Path Iength: 12.8166 A^-1", "Pathlength 12.8166", or "Path length: 12,8166".
    """
    if not text:
        return None

    cleaned = str(text)
    replacements = {
        "Iength": "length",
        "lenth": "length",
        "Length": "length",
        "PathIength": "Path length",
        "Pathlength": "Path length",
        "pathlength": "Path length",
        "：": ":",
        ",": ".",
        "O": "0",
        "o": "0",
    }
    for a, b in replacements.items():
        cleaned = cleaned.replace(a, b)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # v8：删除无锚点的兜底 pattern（原 r"([0-9]+\.[0-9]+)"）——第三方图
    # 通常没有 Path length 标注，兜底会把 y 轴刻度等无关数字静默当作
    # 路径总长（曲率按其平方错），只接受含 Path/length 上下文的匹配。
    patterns = [
        r"Path\s*length[^0-9\-+]*([0-9]+(?:\.[0-9]+)?)",
        r"length[^0-9\-+]*([0-9]+(?:\.[0-9]+)?)",
        r"Path[^0-9\-+]*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, cleaned, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            value = float(m.group(1))
        except Exception:
            continue
        if np.isfinite(value) and 0.05 < value < 200.0:
            return value
    return None


def ocr_path_length_from_image(
    img_bgr: np.ndarray,
    tesseract_cmd: Optional[str] = None,
    ocr_region_out: Optional[Path] = None,
) -> Tuple[Optional[float], str, str]:
    """
    OCR the fixed upper-right "Path length: xxxx Å^-1" label.

    This version is more tolerant of different margins and OCR variants:
    it tries multiple upper-right crop windows, multiple threshold strategies,
    and both unrestricted OCR and character-whitelist OCR.
    """
    try:
        import pytesseract  # type: ignore
    except Exception as exc:
        return None, "image_ocr_unavailable", f"pytesseract unavailable: {exc}"

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.name == "nt" and Path(default_win).exists():
            pytesseract.pytesseract.tesseract_cmd = default_win

    h, w = img_bgr.shape[:2]
    crops = [
        ("top_right_55_18", img_bgr[0:int(0.18 * h), int(0.55 * w):w]),
        ("top_right_45_20", img_bgr[0:int(0.20 * h), int(0.45 * w):w]),
        ("top_right_35_24", img_bgr[0:int(0.24 * h), int(0.35 * w):w]),
        ("top_right_25_28", img_bgr[0:int(0.28 * h), int(0.25 * w):w]),
        ("top_band_full", img_bgr[0:int(0.18 * h), 0:w]),
    ]

    all_texts: List[str] = []
    whitelist = "Pathlength:0123456789. -^AÅa"
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 7",
        "--oem 3 --psm 11",
        f"--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist}",
        f"--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}",
    ]

    for name, crop in crops:
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if ocr_region_out is not None:
            ocr_region_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ocr_region_out.with_name(ocr_region_out.stem + f"_{name}.png")), crop)

        variants = [
            ("gray", gray),
            ("binary190", cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]),
            ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ]
        for vname, g0 in variants:
            for scale in [2, 3, 4, 5]:
                g = cv2.resize(g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                for config in configs:
                    try:
                        txt = pytesseract.image_to_string(g, config=config)
                    except Exception as exc:
                        return None, "image_ocr_error", str(exc)
                    txt = txt.strip()
                    if not txt:
                        continue
                    all_texts.append(f"{name}/{vname}/x{scale}: {txt}")
                    value = parse_path_length_from_text(txt)
                    if value is not None:
                        return value, "image_ocr", txt

    return None, "image_ocr_failed", " | ".join(all_texts[:12])


def resolve_path_length(
    img_bgr: np.ndarray,
    material_id: str,
    path_lengths: Dict[str, float],
    ktotal_arg: Optional[float],
    tesseract_cmd: Optional[str],
    ocr_region_dir: Optional[Path],
) -> Tuple[Optional[float], str, str]:
    ocr_region_out = None
    if ocr_region_dir is not None:
        ocr_region_out = ocr_region_dir / f"{material_id}_ocr_region.png"
    if material_id in path_lengths:
        return path_lengths[material_id], "path_length_table_or_generated", ""
    val, src, raw = ocr_path_length_from_image(
        img_bgr,
        tesseract_cmd=tesseract_cmd,
        ocr_region_out=ocr_region_out,
    )
    if val is not None:
        return val, src, raw
    if ktotal_arg is not None and ktotal_arg > 0:
        return float(ktotal_arg), "command_line_ktotal", raw
    return None, src, raw


def auto_detect_plot_box(img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    dark = gray < 70
    row_counts = dark.sum(axis=1)
    row_candidates = np.where(row_counts > 0.35 * w)[0]
    if len(row_candidates) >= 2:
        top = int(row_candidates.min())
        bottom = int(row_candidates.max())
    else:
        top = int(0.09 * h)
        bottom = int(0.96 * h)
    roi = dark[top:bottom + 1, :]
    col_counts = roi.sum(axis=0)
    roi_h = max(1, bottom - top + 1)
    col_candidates = np.where(col_counts > 0.45 * roi_h)[0]
    if len(col_candidates) >= 2:
        left = int(col_candidates.min())
        right = int(col_candidates.max())
    else:
        left = int(0.12 * w)
        right = int(0.97 * w)
    inset = 3
    return max(0, left + inset), max(0, top + inset), min(w - 1, right - inset), min(h - 1, bottom - inset)


def parse_crop(crop: Optional[str], img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    if crop is None or str(crop).strip().lower() == "auto":
        return auto_detect_plot_box(img_bgr)
    vals = [int(v.strip()) for v in crop.split(",")]
    if len(vals) != 4:
        raise ValueError("--crop should be left,top,right,bottom or auto")
    l, t, r, b = vals
    h, w = img_bgr.shape[:2]
    return max(0, l), max(0, t), min(w - 1, r), min(h - 1, b)


@dataclass
class PiecewiseKmap:
    """归一化横坐标 q∈[0,1] 与物理波矢 k（Å^-1）的分段线性双向映射。

    节点由图像检测到的段边界位置（q_nodes）与结构侧各段边界累积
    k 坐标（k_nodes）一一配对；全局均匀映射是只有两个节点的特例。
    """

    q_nodes: np.ndarray
    k_nodes: np.ndarray
    mode: str

    def k(self, q):
        return np.interp(np.asarray(q, dtype=float), self.q_nodes, self.k_nodes)

    def q(self, k):
        return np.interp(np.asarray(k, dtype=float), self.k_nodes, self.q_nodes)

    @property
    def total_length(self) -> float:
        return float(self.k_nodes[-1] - self.k_nodes[0])


def detect_vertical_separators(
    crop_bgr: np.ndarray,
    min_coverage: float = 0.25,
    edge_margin_frac: float = 0.01,
) -> List[float]:
    """在绘图区内检测高对称点处的灰色竖直分隔线，返回归一化 q 位置。

    分隔线是低饱和、中等亮度的灰色虚线；蓝色能带（高饱和）与黑色
    图框（低亮度）都不会落入该颜色窗口，水平的费米虚线每列只贡献
    1-2 个像素，远低于覆盖率阈值，因此不会误报。
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.int32)
    V = hsv[:, :, 2].astype(np.int32)
    grayish = (S < 60) & (V > 95) & (V < 205)
    h, w = grayish.shape
    if h == 0 or w == 0:
        return []
    coverage = grayish.sum(axis=0) / float(max(1, h))
    margin = max(2, int(edge_margin_frac * w))
    cols = np.where(coverage > min_coverage)[0]
    cols = cols[(cols >= margin) & (cols <= w - 1 - margin)]
    if cols.size == 0:
        return []
    groups: List[List[int]] = [[int(cols[0])]]
    for c in cols[1:]:
        if int(c) - groups[-1][-1] <= 3:
            groups[-1].append(int(c))
        else:
            groups.append([int(c)])
    centers: List[float] = []
    for g in groups:
        centers.append(float(np.average(g, weights=coverage[g])) / float(max(w - 1, 1)))
    return sorted(centers)


def build_kmap(
    seg_info: Optional[Dict[str, object]],
    detected_q: List[float],
    fallback_total: Optional[float],
    q_pad: float = 0.0,
) -> Tuple[Optional[PiecewiseKmap], Dict[str, object]]:
    """组合"结构侧段长 + 图像侧分隔线位置"得到 q->k 映射。

    q_pad：auto 裁剪把绘图区向内缩了 inset 像素，坐标轴真实首末
    高对称点位于归一化坐标 -q_pad 与 1+q_pad 处；所有端点节点都
    以此锚定，避免误差集中到首末两段。

    优先级：
    1. 单段路径——两节点均匀映射（single_segment_uniform）；
    2. 有段信息、检测数量恰为 段数-1 且（自渲染图）位置与等比预期
       偏差小于阈值——逐段标定（per_segment_detected）；数量对但
       位置偏差过大视为配对错位，降级到等比放置并标注；
    3. 有段信息但数量不符——按段长比例放置边界（对按累积 k 坐标
       等比绘制的标准图与逐段检测等价）；
    4. 只有全局总长（外部 CSV / OCR / --ktotal 回退）——均匀映射；
    5. 什么都没有——返回 None，由调用方筛除。
    """
    q_lo, q_hi = -float(q_pad), 1.0 + float(q_pad)
    diag: Dict[str, object] = {
        "n_segments_expected": "",
        "n_boundaries_detected": len(detected_q),
        "boundary_detect_max_err_qnorm": "",
        "k_calibration_mode": "unavailable",
    }
    if seg_info:
        boundaries = np.asarray(seg_info.get("boundaries_Ainv", []), dtype=float)
        if boundaries.size >= 2 and np.all(np.diff(boundaries) >= -1e-12):
            k_nodes = boundaries - boundaries[0]
            total = float(k_nodes[-1])
            if total > 1e-9:
                n_seg = int(len(k_nodes) - 1)
                diag["n_segments_expected"] = n_seg
                expected_q = q_lo + (k_nodes / total) * (q_hi - q_lo)
                interior_expected = expected_q[1:-1]
                if n_seg == 1:
                    diag["k_calibration_mode"] = "single_segment_uniform"
                    return PiecewiseKmap(np.array([q_lo, q_hi]), k_nodes, "single_segment_uniform"), diag
                if len(detected_q) == n_seg - 1:
                    q_nodes = np.concatenate([[q_lo], np.asarray(detected_q, dtype=float), [q_hi]])
                    max_err = float(np.max(np.abs(np.asarray(detected_q) - interior_expected)))
                    diag["boundary_detect_max_err_qnorm"] = max_err
                    # 等比位置校验只对本脚本自渲染图有效（横轴按累积 k
                    # 等比绘制）；外部图（手工提供段信息）各段宽度可能
                    # 不等比，跳过该校验，仅要求节点严格递增。
                    auto_source = str(seg_info.get("source", "")) in {
                        "dat_klabels", "dat_total_only", "hdf5_route"
                    }
                    position_ok = (not auto_source) or max_err <= 0.02
                    if np.all(np.diff(q_nodes) > 1e-6) and position_ok:
                        diag["k_calibration_mode"] = "per_segment_detected"
                        return PiecewiseKmap(q_nodes, k_nodes, "per_segment_detected"), diag
                    diag["k_calibration_mode"] = "per_segment_proportional_position_mismatch"
                    return PiecewiseKmap(expected_q, k_nodes, "per_segment_proportional"), diag
                diag["k_calibration_mode"] = "per_segment_proportional"
                return PiecewiseKmap(expected_q, k_nodes, "per_segment_proportional"), diag
    if fallback_total is not None and np.isfinite(fallback_total) and fallback_total > 0:
        diag["k_calibration_mode"] = "global_uniform"
        return (
            PiecewiseKmap(np.array([q_lo, q_hi]), np.array([0.0, float(fallback_total)]), "global_uniform"),
            diag,
        )
    return None, diag


def make_band_mask(crop_bgr: np.ndarray, mode: str, dark_threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark_mask = gray < dark_threshold
    blue_mask = ((H >= 95) & (H <= 135) & (S > 60) & (V > 40))
    if mode == "blue_dark":
        mask = blue_mask | dark_mask
    elif mode == "blue":
        mask = blue_mask
    elif mode == "dark":
        mask = dark_mask
    else:
        raise ValueError("mode must be blue_dark, blue, or dark")
    red_mask = (((H < 10) | (H > 170)) & (S > 80) & (V > 80))
    mask = mask & (~red_mask)
    border = 4
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    # v8：去噪由 2x2 形态学开运算改为连通域面积过滤。偶数核开运算在
    # 倾斜细线上有方向性偏移（腐蚀只保留与相邻列暗行重叠的部分：上升
    # 段整体偏下、下降段偏上），会造成带边两侧能量的系统性反向偏差，
    # 低分辨率图像上可放大为 ~20% 的曲率误差。面积过滤无方向偏差。
    mask_u8 = mask.astype(np.uint8)
    n_comp, lab, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros(n_comp, dtype=bool)
    if n_comp > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
    return keep[lab]


def cluster_rows(rows: np.ndarray, max_gap: int = 2) -> List[float]:
    if rows.size == 0:
        return []
    rows = np.sort(rows)
    groups: List[List[int]] = [[int(rows[0])]]
    for r in rows[1:]:
        r = int(r)
        if r - groups[-1][-1] <= max_gap:
            groups[-1].append(r)
        else:
            groups.append([r])
    return [float(np.mean(g)) for g in groups]


def extract_points(img_bgr: np.ndarray, crop_box: Tuple[int, int, int, int], emin: float, emax: float, mode: str, dark_threshold: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    l, t, r, b = crop_box
    crop = img_bgr[t:b + 1, l:r + 1].copy()
    h, w = crop.shape[:2]
    mask = make_band_mask(crop, mode=mode, dark_threshold=dark_threshold)
    rows_out = []
    for x in range(w):
        ys = np.where(mask[:, x])[0]
        for y in cluster_rows(ys, max_gap=2):
            q = x / max(w - 1, 1)
            E = emax - y / max(h - 1, 1) * (emax - emin)
            rows_out.append((q, E, x, y, l + x, t + y))
    points = pd.DataFrame(rows_out, columns=["k_norm", "E", "x", "y", "x_abs", "y_abs"])
    return points, crop, mask


def build_envelope(points: pd.DataFrame, kind: str, fermi: float, margin: float) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=points.columns)
    out = []
    for _, group in points.groupby("x"):
        if kind == "cbm":
            cand = group[group["E"] > fermi + margin]
            if not cand.empty:
                out.append(cand.loc[cand["E"].idxmin()].to_dict())
        elif kind == "vbm":
            cand = group[group["E"] < fermi - margin]
            if not cand.empty:
                out.append(cand.loc[cand["E"].idxmax()].to_dict())
        else:
            raise ValueError("kind must be cbm or vbm")
    return pd.DataFrame(out)


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
        # v8：剔除近水平细线（如文献图中深色的 E=0 虚线——面积过滤不再
        # 像旧 2x2 开运算那样顺带删掉 1px 细线）：同一像素行覆盖大量
        # k 列的近费米点不计入金属性穿越统计。
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


def write_metal_debug(
    img_bgr: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    points: pd.DataFrame,
    out_path: Path,
    emin: float,
    emax: float,
    fermi: float,
    fermi_crossing_tol_eV: float,
    title: str,
):
    """Write a debug image for metallic screen-out cases without CBM/VBM fits."""
    out = img_bgr.copy()
    l, t, r, b = crop_box
    cv2.rectangle(out, (l, t), (r, b), (0, 0, 0), 1)
    if points is not None and not points.empty:
        near = points[np.abs(points["E"] - fermi) <= fermi_crossing_tol_eV]
        step = max(1, len(near) // 1200) if not near.empty else 1
        for _, row in near.iloc[::step].iterrows():
            cv2.circle(out, (int(row["x_abs"]), int(row["y_abs"])), 2, (0, 140, 255), -1, cv2.LINE_AA)
    y_f = energy_to_y(fermi, crop_box, emin, emax)
    cv2.line(out, (l, y_f), (r, y_f), (0, 0, 255), 2, cv2.LINE_AA)
    panel_h = 96
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, panel_h), (255, 255, 255), -1)
    cv2.putText(out, title[:150], (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(out, "Metal/semimetal detected: skip CBM/VBM curvature fitting", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 180), 2, cv2.LINE_AA)
    cv2.putText(out, "Orange pixels: band pixels near Fermi level; red line: Fermi level", (15, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


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
        dE_line = np.zeros_like(dE)

    # Local finite-difference slopes as auxiliary dispersion descriptors.
    if len(df) >= 2:
        k_sorted = dk
        E_sorted = df["E"].to_numpy(dtype=float)
        dks = np.diff(k_sorted)
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

    # v8：带边极值点落在段边界 snap 容差内时吸附到边界节点。带边位于
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

    # v8：左右拟合窗口在带边所在段的边界处截断，避免把方向在高对称点
    # 转折的相邻段（不同晶向的分支）混进同一条固定顶点抛物线。带边点
    # 恰好落在段边界上时（snap 容差内），左右两侧各自取相邻段。
    # 截断边界只采用距带边至少 ~10 列的节点：更近的节点若充当边界会
    # 产生几个像素宽的碎片拟合窗口（k 跨度过小，曲率数值不稳定），
    # 此时跳过该节点、窗口延伸到下一节点（稳定的 v7 式行为）。
    nodes = np.asarray(kmap.q_nodes, dtype=float)
    snap = max(vertex_snap_tol, 10.0 * col_dq, 1e-6)
    lower_nodes = nodes[nodes < edge_k - snap]
    upper_nodes = nodes[nodes > edge_k + snap]
    left_bound = float(lower_nodes.max()) if lower_nodes.size else -np.inf
    right_bound = float(upper_nodes.min()) if upper_nodes.size else np.inf

    left_df = local[(local["k_norm"] < edge_k - 1e-9) & (local["k_norm"] >= left_bound)].copy()
    right_df = local[(local["k_norm"] > edge_k + 1e-9) & (local["k_norm"] <= right_bound)].copy()

    # v8：顶点能量去偏。包络的极值是"逐列取极值再全局取极值"的顺序
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


def energy_to_y(E: float, crop_box: Tuple[int, int, int, int], emin: float, emax: float) -> int:
    l, t, r, b = crop_box
    h = max(1, b - t + 1)
    y = t + (emax - E) / (emax - emin) * (h - 1)
    return int(round(np.clip(y, t, b)))


def k_to_x(k_norm: float, crop_box: Tuple[int, int, int, int]) -> int:
    l, t, r, b = crop_box
    w = max(1, r - l + 1)
    x = l + k_norm * (w - 1)
    return int(round(np.clip(x, l, r)))


def draw_points(img: np.ndarray, df: pd.DataFrame, color: Tuple[int, int, int], radius: int = 1, step: int = 1):
    if df is None or df.empty:
        return
    for _, row in df.iloc[::step].iterrows():
        cv2.circle(img, (int(row["x_abs"]), int(row["y_abs"])), radius, color, -1, cv2.LINE_AA)


def draw_curve(img: np.ndarray, curve: pd.DataFrame, crop_box: Tuple[int, int, int, int], emin: float, emax: float, color: Tuple[int, int, int], thickness: int = 2):
    if curve is None or curve.empty:
        return
    pts = []
    for _, row in curve.iterrows():
        x = k_to_x(float(row["k_norm"]), crop_box)
        y = energy_to_y(float(row["E"]), crop_box, emin, emax)
        pts.append([x, y])
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def write_debug(
    img_bgr: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    cbm: Optional[EdgeResult],
    vbm: Optional[EdgeResult],
    out_path: Path,
    emin: float,
    emax: float,
    title: str,
    separators_q: Optional[List[float]] = None,
):
    out = img_bgr.copy()
    l, t, r, b = crop_box
    cv2.rectangle(out, (l, t), (r, b), (0, 0, 0), 1)

    # v8: 图像检测到的高对称点分隔线位置（橙色），用于人工核对逐段标定。
    for q in (separators_q or []):
        x_sep = k_to_x(float(q), crop_box)
        cv2.line(out, (x_sep, t), (x_sep, b), (0, 140, 255), 1, cv2.LINE_AA)

    # Colors in BGR
    colors = {
        ("cbm", "left"): (0, 0, 255),       # red
        ("cbm", "right"): (255, 0, 255),    # magenta
        ("vbm", "left"): (255, 255, 0),     # cyan
        ("vbm", "right"): (0, 180, 0),      # green
    }

    for edge in [cbm, vbm]:
        if edge is None:
            continue
        # sparse envelope
        env_color = (180, 180, 180)
        step = max(1, len(edge.envelope) // 350) if not edge.envelope.empty else 1
        draw_points(out, edge.envelope, env_color, radius=1, step=step)
        # edge marker
        x = k_to_x(edge.edge_k_norm, crop_box)
        y = energy_to_y(edge.edge_E, crop_box, emin, emax)
        marker_color = (0, 0, 255) if edge.kind == "cbm" else (255, 80, 0)
        cv2.circle(out, (x, y), 8, marker_color, 2, cv2.LINE_AA)
        cv2.putText(out, edge.kind.upper(), (x + 8, y - 8 if edge.kind == "cbm" else y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, marker_color, 2, cv2.LINE_AA)
        # side points and curves
        for side_name, side_fit in [("left", edge.left), ("right", edge.right)]:
            color = colors[(edge.kind, side_name)]
            if side_fit is None or side_fit.side_points.empty:
                continue
            draw_points(out, side_fit.side_points, color, radius=2, step=1)
            draw_curve(out, side_fit.quad_curve, crop_box, emin, emax, color, thickness=2)
            # Optional thin through-vertex linear line to show near-linear diagnosis.
            draw_curve(out, side_fit.line_curve, crop_box, emin, emax, color, thickness=1)

    panel_h = 92
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, panel_h), (255, 255, 255), -1)
    cv2.putText(out, title[:150], (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(out, "CBM-L red, CBM-R magenta; VBM-L cyan, VBM-R green; thick=quadratic, thin=line", (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(out, "Fixed-vertex quadratic fit: E-E0=A(k-k0)^2, curvature=2A", (15, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def opt(v: Optional[float]) -> object:
    if v is None:
        return ""
    try:
        if not np.isfinite(v):
            return ""
    except Exception:
        return v
    return float(v)


def _finite_float(v: Optional[float]) -> Optional[float]:
    """Return float(v) if finite, otherwise None."""
    if v is None:
        return None
    try:
        vf = float(v)
    except Exception:
        return None
    if not np.isfinite(vf):
        return None
    return vf


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
    Evaluate material-level screening flags.

    Band-gap selection:
        selected only if the image-estimated gap is finite and in
        [gap_min_eV, gap_max_eV].

    Curvature selection:
        selected if any CBM/VBM side is sign-correct and its absolute
        curvature lies inside [curvature_min_eVA2, curvature_max_eVA2].

    Linear-like high-dispersion selection:
        retained as the original auxiliary branch for near-linear bands.

    Final selected:
        gap_selected AND (curvature_selected OR linear_high_dispersion_selected).
        Direct-gap filtering, when enabled, is applied later by
        apply_direct_gap_screening().
    """
    side_fits = _collect_side_fits(cbm, vbm)

    gap_val = _finite_float(image_gap_eV)
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
        C = _finite_float(sf.curvature_eVA2)
        mean_slope = _finite_float(sf.mean_abs_slope_eVA)
        max_slope = _finite_float(sf.max_abs_slope_eVA)

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


def process_one(image_path: Path, args: argparse.Namespace, path_lengths: Dict[str, float], segments_map: Dict[str, dict], output_debug_dir: Path, ocr_region_dir: Optional[Path]) -> Dict[str, object]:
    """
    Process one standardized band image.

    v8: the pixel->Å^-1 mapping is piecewise per k-path segment, built from
    (a) structure-side segment boundaries in segments_map and (b) the gray
    vertical separator lines detected in the image itself. Images without
    segment metadata fall back to the v7 global chain (CSV / OCR / --ktotal).

    v5 logic:
    - If no band pixels are visible in [-1, 1] eV, screen out normally.
    - If CBM/VBM envelope cannot be constructed, screen out normally.
    - If CBM/VBM can be constructed and Path length can be read, always calculate
      all available left/right curvature and slope descriptors, regardless of
      whether the gap is inside 0.1–0.5 eV.
    - The gap condition is only used in final_selected, not for deciding whether
      curvature should be output.
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
    # 信息的图像回退到 v7 的全局链（外部 CSV / OCR / --ktotal）。
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
        k_total, path_src, path_raw = resolve_path_length(
            img, material_id, path_lengths, args.ktotal, args.tesseract_cmd, ocr_region_dir
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
        try:
            row.update(metal_info)
        except NameError:
            pass
        return row

    cbm = fit_edge_sides(
        cbm_env,
        "cbm",
        args.local_width,
        args.energy_window,
        kmap,
        args.min_side_points,
        args.rmse_threshold_eV,
        args.min_quadratic_span_eV,
        args.linear_preference_tol,
    )
    vbm = fit_edge_sides(
        vbm_env,
        "vbm",
        args.local_width,
        args.energy_window,
        kmap,
        args.min_side_points,
        args.rmse_threshold_eV,
        args.min_quadratic_span_eV,
        args.linear_preference_tol,
    )

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
    try:
        row.update(metal_info)
    except NameError:
        pass
    row.update(kdiag)
    # v8：CBM-VBM 的 Å^-1 距离按分段映射计算（apply_direct_gap_screening
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



# =====================================================================
# Raw DAT / HDF5 -> standardized band-image preprocessing
# =====================================================================

HARTREE_TO_EV = 27.211386245988


def _safe_plot_name(name: str) -> str:
    name = str(name).strip()
    lower = name.lower()
    if lower.startswith("poscar-") or lower.startswith("poscar_"):
        name = name[7:]
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", "-", name).strip("-")
    return name or "band"


def _reserve_unique_stem(base: str, used: set, hint: str = "") -> str:
    base = _safe_plot_name(base)
    if base not in used:
        used.add(base)
        return base
    candidate = f"{base}__{hint}" if hint else f"{base}__2"
    counter = 2
    while candidate in used:
        candidate = f"{base}__{hint}_{counter}" if hint else f"{base}__{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _get_matplotlib_pyplot():
    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    return plt


def _format_klabel(label: str) -> str:
    label = str(label).strip()
    upper = label.upper()
    if upper in {"G", "GAMMA", "Γ"}:
        return r"$\Gamma$"
    if "_" in label:
        i = label.find("_")
        return label[:i] + "$" + label[i:i + 2] + "$" + label[i + 2:]
    return label


def _read_optional_klabels(klabels_file: Path, x_shift: float = 0.0) -> Tuple[List[float], List[str]]:
    if not klabels_file.exists():
        return [], []
    ticks: List[float] = []
    labels: List[str] = []
    try:
        lines = klabels_file.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]
    except Exception:
        return [], []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        parts = s.split()
        if len(parts) != 2:
            continue
        try:
            xpos = float(parts[1]) - float(x_shift)
        except Exception:
            continue
        ticks.append(xpos)
        labels.append(_format_klabel(parts[0]))
    return ticks, labels


def _standard_band_plot(
    x_axis: np.ndarray,
    energies: np.ndarray,
    output_png: Path,
    path_length: float,
    emin: float,
    emax: float,
    dpi: int = 300,
    line_width: float = 1.0,
    xticks: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> None:
    """Plot the standardized image consumed by the image-recognition workflow."""
    plt = _get_matplotlib_pyplot()
    x_axis = np.asarray(x_axis, dtype=float).reshape(-1)
    energies = np.asarray(energies, dtype=float)
    if energies.ndim == 1:
        energies = energies[:, np.newaxis]
    if energies.ndim != 2:
        raise ValueError(f"energies must be 2-D (nk, nbands), got {energies.shape}")
    if len(x_axis) != energies.shape[0]:
        raise ValueError(f"x-axis length {len(x_axis)} != energy rows {energies.shape[0]}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(y=0, xmin=0, xmax=1, linestyle="--", linewidth=0.5, color="0.5")

    ticks = list(xticks or [])
    labs = list(labels or [])
    for tick in ticks[1:-1]:
        ax.axvline(x=tick, ymin=0, ymax=1, linestyle="--", linewidth=0.5, color="0.5")

    # All bands are blue so the downstream HSV segmentation is deterministic.
    for iband in range(energies.shape[1]):
        y = energies[:, iband]
        if not np.any(np.isfinite(y)):
            continue
        if np.nanmax(y) < emin - 0.5 or np.nanmin(y) > emax + 0.5:
            continue
        ax.plot(x_axis, y, linewidth=line_width, color="blue")

    ax.set_ylabel(r"$\mathrm{Energy}$ (eV)", fontsize=15)
    ax.set_xlim((float(np.nanmin(x_axis)), float(np.nanmax(x_axis))))
    ax.set_ylim((emin, emax))
    ax.tick_params(axis="y", labelsize=13)

    if ticks and len(ticks) == len(labs):
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs, fontsize=13)
    else:
        ax.set_xticks([])

    fig.text(
        0.985, 0.975,
        f"Path length: {path_length:.4f} Å^-1",
        ha="right", va="top", fontsize=11, fontname="DejaVu Sans",
        bbox=dict(facecolor="white", edgecolor="black",
                  boxstyle="round,pad=0.25", linewidth=0.5),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def _dat_material_name(dat_file: Path) -> str:
    poscar = dat_file.parent / "POSCAR"
    if poscar.exists():
        try:
            lines = poscar.read_text(encoding="utf-8", errors="ignore").splitlines()
            if lines and lines[0].strip():
                return _safe_plot_name(lines[0].strip())
        except Exception:
            pass
    if dat_file.stem.lower() in {"band", "bands", "bandstructure"}:
        return _safe_plot_name(dat_file.parent.name)
    return _safe_plot_name(dat_file.stem)


def find_dat_band_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    return sorted(input_dir.rglob("*.dat"))


def _segment_boundaries(ticks: List[float], total_length: float) -> List[float]:
    """把 KLABELS/route 刻度整理成单调的段边界累积坐标 [0, ..., L]（Å^-1）。"""
    total = float(total_length)
    vals = [float(t) for t in (ticks or []) if np.isfinite(t)]
    vals = sorted(min(max(v, 0.0), total) for v in vals)
    tol = max(1e-9, 1e-6 * total)
    merged: List[float] = []
    for v in vals:
        if not merged or v - merged[-1] > tol:
            merged.append(v)
    if not merged or merged[0] > tol:
        merged.insert(0, 0.0)
    if abs(merged[-1] - total) > tol:
        merged.append(total)
    return merged


def convert_dat_band_files(
    input_dir: Path,
    output_dir: Path,
    emin: float,
    emax: float,
    dpi: int,
    line_width: float,
    used_stems: set,
) -> Tuple[Dict[str, float], Dict[str, dict], Dict[str, int]]:
    """
    DAT format:
      first column = cumulative k-path coordinate (Å^-1)
      columns 2...N = band energies (eV)

    Optional KLABELS/POSCAR in the same directory are used when available.
    """
    files = find_dat_band_files(input_dir)
    path_lengths: Dict[str, float] = {}
    segments_map: Dict[str, dict] = {}
    stats = {"found": len(files), "success": 0, "failed": 0}

    for idx, dat_file in enumerate(files, start=1):
        print(f"[DAT {idx}/{len(files)}] {dat_file}")
        try:
            arr = np.loadtxt(dat_file, dtype=float)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 3:
                raise ValueError("DAT must contain >=3 rows and >=2 columns")

            x_raw = np.asarray(arr[:, 0], dtype=float)
            energies = np.asarray(arr[:, 1:], dtype=float)
            x_min = float(np.nanmin(x_raw))
            x_max = float(np.nanmax(x_raw))
            path_length = x_max - x_min
            if not np.isfinite(path_length) or path_length <= 1e-10:
                raise ValueError("Invalid/zero k-path length")

            x_axis = x_raw - x_min
            ticks, labels = _read_optional_klabels(dat_file.parent / "KLABELS", x_shift=x_min)

            base = _dat_material_name(dat_file)
            stem = _reserve_unique_stem(base, used_stems, hint="dat")
            output_png = output_dir / f"{stem}.png"

            _standard_band_plot(
                x_axis=x_axis, energies=energies, output_png=output_png,
                path_length=path_length, emin=emin, emax=emax,
                dpi=dpi, line_width=line_width, xticks=ticks, labels=labels,
            )
            path_lengths[safe_stem(stem)] = float(path_length)
            aligned = bool(ticks) and len(labels) == len(ticks)
            segments_map[safe_stem(stem)] = {
                "labels": [str(l) for l in labels] if aligned else [],
                "boundaries_Ainv": _segment_boundaries(ticks, path_length),
                # 原始 (标签, 累积k) 刻度对；boundaries_Ainv 经过合并/补插，
                # 与 labels 可能不再逐位对应，审计请以 ticks_raw 为准。
                "ticks_raw": [[str(l), float(t)] for l, t in zip(labels, ticks)] if aligned else [],
                "source": "dat_klabels" if ticks else "dat_total_only",
            }
            stats["success"] += 1
            print(f"  OK -> {output_png.name} | path length={path_length:.4f} Å^-1")
        except Exception as exc:
            stats["failed"] += 1
            print(f"  ERROR: {exc}")

    return path_lengths, segments_map, stats


# -------------------- QuantumATK/ATK HDF5 support --------------------

def _decode_h5_scalar(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if hasattr(x, "decode"):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _h5_read_unit(group, path: str) -> str:
    try:
        return _decode_h5_scalar(group[path][()])
    except Exception:
        return ""


def _unit_factor_to_ev(unit: str) -> float:
    u = str(unit).lower().strip()
    if u in {"hartree", "ha"}:
        return HARTREE_TO_EV
    if u in {"ev", "electronvolt", "electron_volt"}:
        return 1.0
    return 1.0


def _available_band_groups(h5) -> List[str]:
    groups = [k for k in h5.keys() if re.match(r"Bandstructure_\d+$", str(k))]
    groups.sort(key=lambda s: int(str(s).split("_")[-1]))
    return groups


def _select_band_group(h5, requested: Optional[str]) -> str:
    groups = _available_band_groups(h5)
    if not groups:
        raise RuntimeError("No Bandstructure_* group found")
    if requested:
        if requested not in h5:
            raise RuntimeError(f"Requested group {requested!r} not found; available={groups}")
        return requested
    return groups[-1]


def _read_h5_route(bg) -> List[Tuple[str, str]]:
    route_group = bg["BaseBandstructure/route"]
    keys = sorted(route_group.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    route: List[Tuple[str, str]] = []
    for key in keys:
        g = route_group[key]
        route.append((_decode_h5_scalar(g["0/data"][()]), _decode_h5_scalar(g["1/data"][()])))
    return route


def _read_h5_lattice_angstrom(bg) -> np.ndarray:
    base = "BaseBandstructure/lattice/BravaisLattice/primitive_vectors"
    vectors = np.asarray(bg[f"{base}/array/data"][()], dtype=float)
    unit = _h5_read_unit(bg, f"{base}/unit/data")
    if unit.lower().strip() in {"bohr", "a0"}:
        vectors = vectors * 0.529177210903
    return vectors


def _reciprocal_vectors_from_lattice(a_vectors: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(a_vectors).T


def _compute_h5_kpath_axis(
    frac_path: np.ndarray,
    reciprocal_vectors: np.ndarray,
    route: List[Tuple[str, str]],
) -> Tuple[np.ndarray, List[float], List[str]]:
    npts = len(frac_path)
    nseg = len(route)

    if nseg <= 0:
        cart = frac_path @ reciprocal_vectors
        x = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cart, axis=0), axis=1))]
        return x, [float(x[0]), float(x[-1])], ["", ""]

    if (npts - 1) % nseg != 0:
        cart = frac_path @ reciprocal_vectors
        x = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cart, axis=0), axis=1))]
        return x, [float(x[0]), float(x[-1])], [
            _format_klabel(route[0][0]), _format_klabel(route[-1][1])
        ]

    points_per_segment = (npts - 1) // nseg
    x = np.zeros(npts, dtype=float)
    ticks: List[float] = [0.0]
    labels_raw: List[str] = [route[0][0]]
    cumulative = 0.0

    for iseg in range(nseg):
        start_idx = iseg * points_per_segment
        end_idx = (iseg + 1) * points_per_segment
        if start_idx > 0:
            x[start_idx] = cumulative
        for ip in range(start_idx + 1, end_idx + 1):
            dk_frac = frac_path[ip] - frac_path[ip - 1]
            dk_cart = dk_frac @ reciprocal_vectors
            cumulative += float(np.linalg.norm(dk_cart))
            x[ip] = cumulative
        ticks.append(cumulative)
        end_label = route[iseg][1]
        if iseg + 1 < nseg:
            next_start = route[iseg + 1][0]
            if next_start != end_label:
                end_label = f"{end_label}|{next_start}"
        labels_raw.append(end_label)

    return x, ticks, [_format_klabel(lbl) for lbl in labels_raw]


def _read_hdf5_band_data(
    hdf5_file: Path,
    band_group: Optional[str],
    subtract_fermi: bool,
) -> Dict[str, object]:
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(f"h5py is required for HDF5 conversion: {exc}")

    with h5py.File(hdf5_file, "r") as h5:
        group_name = _select_band_group(h5, band_group)
        bg = h5[group_name]

        eig = np.asarray(bg["eigenvalues/array/data"][()], dtype=float)
        fermi = float(bg["fermi_level/array/data"][()])
        eig_unit = _h5_read_unit(bg, "eigenvalues/unit/data")
        fermi_unit = _h5_read_unit(bg, "fermi_level/unit/data")
        factor = _unit_factor_to_ev(eig_unit)
        fermi_factor = _unit_factor_to_ev(fermi_unit) if fermi_unit else factor

        if eig.ndim == 2:
            eig = eig[np.newaxis, :, :]
        if eig.ndim != 3:
            raise RuntimeError(f"Unsupported eigenvalue shape: {eig.shape}")

        energies_ev = eig * factor
        if subtract_fermi:
            energies_ev = energies_ev - fermi * fermi_factor

        frac_path = np.asarray(bg["BaseBandstructure/path/data"][()], dtype=float)
        lattice = _read_h5_lattice_angstrom(bg)
        reciprocal = _reciprocal_vectors_from_lattice(lattice)
        route = _read_h5_route(bg)
        x_axis, xticks, labels = _compute_h5_kpath_axis(frac_path, reciprocal, route)

    energies_2d = np.transpose(energies_ev, (1, 0, 2)).reshape(
        energies_ev.shape[1], energies_ev.shape[0] * energies_ev.shape[2]
    )
    return {
        "group_name": group_name,
        "energies_2d": energies_2d,
        "x_axis": x_axis,
        "xticks": xticks,
        "labels": labels,
        "path_length": float(x_axis[-1]),
    }


def find_hdf5_band_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    files: List[Path] = []
    files.extend(input_dir.rglob("*.hdf5"))
    files.extend(input_dir.rglob("*.h5"))
    return sorted(set(files))


def convert_hdf5_band_files(
    input_dir: Path,
    output_dir: Path,
    emin: float,
    emax: float,
    dpi: int,
    line_width: float,
    band_group: Optional[str],
    subtract_fermi: bool,
    used_stems: set,
) -> Tuple[Dict[str, float], Dict[str, dict], Dict[str, int]]:
    files = find_hdf5_band_files(input_dir)
    path_lengths: Dict[str, float] = {}
    segments_map: Dict[str, dict] = {}
    stats = {"found": len(files), "success": 0, "failed": 0}

    for idx, hdf5_file in enumerate(files, start=1):
        print(f"[HDF5 {idx}/{len(files)}] {hdf5_file}")
        try:
            data = _read_hdf5_band_data(hdf5_file, band_group, subtract_fermi)
            stem = _reserve_unique_stem(_safe_plot_name(hdf5_file.stem), used_stems, hint="hdf5")
            output_png = output_dir / f"{stem}.png"
            _standard_band_plot(
                x_axis=np.asarray(data["x_axis"]),
                energies=np.asarray(data["energies_2d"]),
                output_png=output_png,
                path_length=float(data["path_length"]),
                emin=emin, emax=emax, dpi=dpi, line_width=line_width,
                xticks=list(data["xticks"]), labels=list(data["labels"]),
            )
            path_lengths[safe_stem(stem)] = float(data["path_length"])
            segments_map[safe_stem(stem)] = {
                "labels": [str(l) for l in data["labels"]],
                "boundaries_Ainv": _segment_boundaries(list(data["xticks"]), float(data["path_length"])),
                "ticks_raw": [
                    [str(l), float(t)] for l, t in zip(data["labels"], data["xticks"])
                ] if len(data["labels"]) == len(data["xticks"]) else [],
                "source": "hdf5_route",
            }
            stats["success"] += 1
            print(
                f"  OK -> {output_png.name} | group={data['group_name']} | "
                f"path length={float(data['path_length']):.4f} Å^-1"
            )
        except Exception as exc:
            stats["failed"] += 1
            print(f"  ERROR: {exc}")

    return path_lengths, segments_map, stats


def prepare_standard_band_images(args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, dict], Dict[str, Dict[str, int]]]:
    """
    Automatically detect and convert both DAT and HDF5/H5 band files from
    ./input into standardized PNG files in ./band_images.

    v8: additionally collects per-material k-path segment metadata
    (high-symmetry labels + cumulative segment boundaries in Å^-1) and
    writes it to band_images/kpath_segments.json. This file is the declared
    crystal-side input of the screening stage; it carries no plot-pixel
    information.

    No input-format selection is required:
        *.dat        -> DAT converter
        *.hdf5/*.h5  -> QuantumATK/ATK HDF5 converter

    Both types may coexist in the same input directory or its subdirectories.
    Existing images directly under band_images are removed first so results
    from a previous batch do not contaminate the current screening run.
    """
    raw_dir = Path(args.raw_input_dir)
    output_dir = Path(args.input)

    if not raw_dir.exists():
        raise SystemExit(f"Raw input folder not found: {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # v8：input 中没有任何原始能带文件时不再直接退出，也不删除现有
    # 图片——允许"只筛选 band_images 中已有图像（含第三方图）"的用法。
    if not find_dat_band_files(raw_dir) and not find_hdf5_band_files(raw_dir):
        print(
            f"WARNING: no .dat/.hdf5/.h5 files in {raw_dir}; "
            f"keeping existing images in {output_dir} and skipping regeneration."
        )
        return {}, {}, {}

    # Rebuild band_images for the current batch only.
    # v8：只删除上一轮自动生成的图片（记录在旧 kpath_segments.json 中），
    # 保留用户手工放入的第三方图；没有旧记录时保持 v7 行为（全部清除）。
    old_json = output_dir / "kpath_segments.json"
    old_auto_stems: Optional[set] = None
    if old_json.exists():
        try:
            old_auto_stems = set(json.loads(old_json.read_text(encoding="utf-8")).keys())
        except Exception:
            old_auto_stems = None
    # kpath_segments_external.json 声明的第三方图无条件保留（键规范化
    # 与消费端一致），即使没有上一轮的自动生成记录。
    preserve_stems: set = set()
    ext_json = output_dir / "kpath_segments_external.json"
    if ext_json.exists():
        try:
            preserve_stems = {
                safe_stem(Path(str(k)).stem)
                for k in json.loads(ext_json.read_text(encoding="utf-8"))
            }
        except Exception:
            preserve_stems = set()
    for pattern in IMAGE_EXTS:
        for old_img in output_dir.glob(pattern):
            stem_n = safe_stem(old_img.stem)
            if stem_n in preserve_stems:
                continue
            if old_auto_stems is not None and stem_n not in old_auto_stems:
                continue
            try:
                old_img.unlink()
            except Exception as exc:
                print(f"WARNING: cannot remove old image {old_img}: {exc}")

    generated_path_lengths: Dict[str, float] = {}
    segments_all: Dict[str, dict] = {}
    summary: Dict[str, Dict[str, int]] = {}
    used_stems: set = set()

    # Always scan DAT files.
    dat_map, dat_segments, dat_stats = convert_dat_band_files(
        input_dir=raw_dir,
        output_dir=output_dir,
        emin=args.emin,
        emax=args.emax,
        dpi=args.standard_plot_dpi,
        line_width=args.standard_plot_line_width,
        used_stems=used_stems,
    )
    generated_path_lengths.update(dat_map)
    segments_all.update(dat_segments)
    summary["DAT"] = dat_stats

    # Always scan HDF5/H5 files.
    h5_map, h5_segments, h5_stats = convert_hdf5_band_files(
        input_dir=raw_dir,
        output_dir=output_dir,
        emin=args.emin,
        emax=args.emax,
        dpi=args.standard_plot_dpi,
        line_width=args.standard_plot_line_width,
        band_group=args.hdf5_bandstructure,
        subtract_fermi=args.hdf5_subtract_fermi,
        used_stems=used_stems,
    )
    generated_path_lengths.update(h5_map)
    segments_all.update(h5_segments)
    summary["HDF5"] = h5_stats

    segments_json = output_dir / "kpath_segments.json"
    try:
        segments_json.write_text(
            json.dumps(segments_all, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"K-path segments metadata: {segments_json} ({len(segments_all)} materials)")
    except Exception as exc:
        print(f"WARNING: cannot write {segments_json}: {exc}")

    total_success = int(dat_stats.get("success", 0)) + int(h5_stats.get("success", 0))

    if total_success == 0:
        raise SystemExit(
            "Raw band files were found, but none could be converted into "
            "standardized band images. Check the DAT/HDF5 file format."
        )

    return generated_path_lengths, segments_all, summary


def write_method_file(out_path: Path):
    text = r"""# 基于能带图像的左右侧二阶曲率、近线性高色散与候选筛选方法

## 1. 目标与适用范围

本程序用于从统一格式的能带图片中自动识别 CBM 和 VBM，并分别在其左右两侧提取带边二阶曲率、图像估算有效质量以及辅助斜率描述符。该方法面向高通量材料筛选中的前置预筛选任务，用于快速识别低有效质量、高迁移率潜力材料。由于输入为能带图片而非原始能带数据，所得曲率和有效质量为图像近似结果，不替代基于原始能带数据的严格有效质量或迁移率计算。

## 2. 图像坐标到物理坐标的转换

所有能带图片统一绘制在能量窗口 \([E_{\min},E_{\max}]=[-1,1]\,\mathrm{eV}\) 内。设绘图区宽度和高度分别为 \(W\) 和 \(H\)，能带像素点的横纵坐标为 \((x_i,y_i)\)。程序将横坐标转换为归一化高对称路径坐标：

\[
q_i=\frac{x_i}{W-1},\qquad q_i\in[0,1].
\]

纵坐标转换为能量：

\[
E_i=E_{\max}-\frac{y_i}{H-1}(E_{\max}-E_{\min}).
\]

若图片右上角标注的整条高对称路径长度为 \(L\)，单位为 \(\mathrm{\AA^{-1}}\)，则实际路径坐标为：

\[
k_i=q_iL.
\]

因此后续所有斜率和曲率均在物理 \(k\) 空间中计算，斜率单位为 \(\mathrm{eV\cdot\AA}\)，二阶曲率单位为 \(\mathrm{eV\cdot\AA^2}\)。

## 3. CBM/VBM 包络识别

在每一个横向像素列，也就是每一个 \(q_j\) 位置，程序从识别到的能带像素中构造导带与价带边缘包络。对于导带，取参考能级 \(E_F\) 上方的最低能量点：

\[
E_{\mathrm{CB}}(q_j)=\min\{E(q_j)\mid E(q_j)>E_F+\delta\}.
\]

对于价带，取参考能级下方的最高能量点：

\[
E_{\mathrm{VB}}(q_j)=\max\{E(q_j)\mid E(q_j)<E_F-\delta\}.
\]

其中 \(\delta\) 是能量间隔参数，用于避免费米能级或参考能级附近的虚线、噪声像素被误识别。

CBM 定义为导带包络的最低点：

\[
(q_c,E_c)=\arg\min_q E_{\mathrm{CB}}(q),
\]

VBM 定义为价带包络的最高点：

\[
(q_v,E_v)=\arg\max_q E_{\mathrm{VB}}(q).
\]

## 4. 左右两侧局域带边窗口

对于识别到的带边点 \((q_0,E_0)\)，程序在其附近选取局域窗口。对于 CBM：

\[
|q_i-q_0|\leq\Delta q,\qquad E_i\leq E_0+\Delta E.
\]

其中 \(\Delta E\) 可在程序运行时交互输入，默认值为 \(0.10\,\mathrm{eV}\)，用于限制拟合点仅来自带边附近的局域分支，减少邻近能带混入。

对于 VBM：

\[
|q_i-q_0|\leq\Delta q,\qquad E_i\geq E_0-\Delta E.
\]

随后以带边点为界，将局域点分为左侧与右侧：

\[
\mathrm{left}: q_i<q_0,
\]

\[
\mathrm{right}: q_i>q_0.
\]

这样可以分别表征高对称路径两侧不同晶向上的带边色散。若 CBM/VBM 位于高对称点，左右两侧通常对应不同高对称方向；若带边位于路径内部，左右两侧则对应同一连续分支的两边。

## 5. 固定顶点二阶曲率拟合

为更接近有效质量定义，程序采用固定顶点二阶拟合，而不是仅计算一阶斜率。对于某一侧 \(\mathrm{side}\)，令：

\[
\Delta k_i=k_i-k_0=L(q_i-q_0),
\]

\[
\Delta E_i=E_i-E_0.
\]

程序采用如下固定顶点二次模型：

\[
\Delta E_i=A_{\mathrm{side}}(\Delta k_i)^2.
\]

该模型强制通过已识别的带边点 \((k_0,E_0)\)，符合带边极值点附近一阶导数趋近于零的物理图像。二次系数通过最小二乘求得：

\[
A_{\mathrm{side}}=\frac{\sum_i (\Delta k_i)^2\Delta E_i}{\sum_i (\Delta k_i)^4}.
\]

对应二阶曲率为：

\[
C_{\mathrm{side}}=\frac{d^2E}{dk^2}=2A_{\mathrm{side}}.
\]

对于 CBM，合理曲率符号应为：

\[
C_{\mathrm{CBM}}>0,
\]

对于 VBM，合理曲率符号应为：

\[
C_{\mathrm{VBM}}<0.
\]

## 6. 图像估算有效质量

有效质量与带边二阶曲率满足：

\[
\frac{1}{m^*}=\frac{1}{\hbar^2}\frac{d^2E}{dk^2}.
\]

当能量单位为 eV、波矢单位为 \(\mathrm{\AA^{-1}}\) 时，有：

\[
\frac{m^*}{m_0}=\frac{7.619964}{|C_{\mathrm{side}}|}.
\]

其中 \(C_{\mathrm{side}}\) 的单位为 \(\mathrm{eV\cdot\AA^2}\)。程序分别输出：

- CBM-left 图像估算电子有效质量；
- CBM-right 图像估算电子有效质量；
- VBM-left 图像估算空穴有效质量；
- VBM-right 图像估算空穴有效质量。

这些结果用于高通量预筛选，不作为最终有效质量数值。

## 7. 斜率辅助描述符

考虑到图像分辨率有限、线宽和能带重叠可能导致二阶曲率拟合不稳定，程序同时保留左右两侧的一阶斜率作为辅助色散描述符。相邻点间斜率为：

\[
s_i=\left|\frac{E_{i+1}-E_i}{k_{i+1}-k_i}\right|.
\]

每一侧输出平均绝对斜率：

\[
\bar{s}_{\mathrm{side}}=\frac{1}{N-1}\sum_{i=1}^{N-1}s_i,
\]

以及最大绝对斜率：

\[
s^{\max}_{\mathrm{side}}=\max_i(s_i).
\]

此外，程序还对每一侧做过带边点的一阶线性模型：

\[
\Delta E_i=B_{\mathrm{side}}\Delta k_i,
\]

其斜率 \(B_{\mathrm{side}}\) 用于判断局域分支是否在图像分辨率下接近线性。

## 8. 可靠性标记

由于输入为图像，曲率估计可能受到像素分辨率、线宽、带交叉和分支识别误差影响。因此程序输出如下可靠性指标：

1. 拟合点数；
2. 曲率符号是否合理；
3. 二次拟合 RMSE；
4. 线性拟合 RMSE；
5. 二次项贡献能量；
6. 是否近似线性；
7. 曲率是否可靠。

其中二次项贡献能量定义为：

\[
\Delta E_{\mathrm{quad}}=|A_{\mathrm{side}}|\max_i|\Delta k_i|^2.
\]

若 \(\Delta E_{\mathrm{quad}}\) 过小，说明在当前图像分辨率下二次项贡献接近噪声水平，曲率结果可能不稳定。若线性模型的误差与二次模型相当或更低，则该分支会被标记为近似线性。最终只有同时满足拟合点数充足、曲率符号合理、二次拟合误差较小且非近似线性的分支，才被标记为曲率可靠。

## 9. 近线性高色散分支的处理

对于普通半导体抛物线型带边，二阶曲率是低有效质量潜力的主描述符。然而，若局域带边在图像分辨率下接近线性，传统抛物线有效质量近似可能不再稳定。此时程序不强行将曲率估算有效质量作为主要依据，而是使用一阶斜率描述高色散特征。

若某一侧满足：

\[
\mathrm{near\_linear}=\mathrm{True}
\]

且平均绝对斜率或最大绝对斜率超过阈值：

\[
\bar{s}_{\mathrm{side}}\ge s_{\mathrm{mean}}^{\mathrm{th}}
\]

或：

\[
s_{\mathrm{side}}^{\max}\ge s_{\max}^{\mathrm{th}},
\]

则该方向被标记为近线性高色散方向。近线性高色散方向不直接使用曲率有效质量作为最终物理结论，而作为具有较高群速度潜力的候选，需要后续基于原始能带数据进一步复核。

## 10. 候选材料筛选规则

本程序同时考虑带隙范围、带边曲率和近线性高色散特征。若在 \([-1,1]\) eV 能量窗口内未识别到能带，或无法构造 CBM/VBM 包络，则样本被直接标记为筛除，而不作为程序错误处理。对于可以构造 CBM/VBM 包络并读取路径长度的图像，程序会输出所有可拟合方向的曲率和斜率描述符；带隙条件仅用于最终候选标记，不再阻断曲率输出。最终筛选中，带隙范围由程序运行时交互输入，默认值为：

\[
0.1\ \mathrm{eV}\le E_g \le 0.5\ \mathrm{eV}.
\]

若启用直接带隙筛选，则程序进一步要求 CBM 与 VBM 的归一化 k 点位置差满足 \(|k_{\mathrm{CBM}}-k_{\mathrm{VBM}}|\leq\Delta k_{\mathrm{direct}}\)，默认 \(\Delta k_{\mathrm{direct}}=0.01\)。在满足带隙条件和直接带隙条件的基础上，若 CBM 或 VBM 的任一左右方向满足：

\[
\kappa_{\min}\le |C_{\mathrm{side}}|\le \kappa_{\max},
\]

且曲率符号符合 CBM/VBM 的物理要求，则材料被标记为曲率筛选命中。若材料存在近线性高色散方向，则被标记为近线性高色散命中。最终候选定义为：

\[
\mathrm{Selected}=\mathrm{GapHit}\ \land\left(\mathrm{CurvatureHit}\ \lor\ \mathrm{LinearHighDispersionHit}\right).
\]

命中候选的 debug 图片会被自动复制到 `selected_debug_images/final_selected` 文件夹中；其中带隙命中的图片会额外复制到 `selected_debug_images/gap_0p1_0p5_semiconductor`，曲率命中的图片会额外复制到 `selected_debug_images/curvature_selected_absC_ge_threshold`，近线性高色散命中的图片会额外复制到 `selected_debug_images/linear_high_dispersion`。

## 11. 推荐表述

可在文章中表述为：

> 为使基于能带图像的输运潜力筛选更符合有效质量的物理定义，本工作在自动识别的 CBM 和 VBM 附近分别进行左右两侧的固定顶点二阶曲率拟合。对于带边点 \((k_0,E_0)\)，局域带边分支按照 \(k_i<k_0\) 与 \(k_i>k_0\) 分为左右两侧，并分别采用 \(E_i-E_0=A_{\mathrm{side}}(k_i-k_0)^2\) 进行拟合。对应曲率为 \(C_{\mathrm{side}}=2A_{\mathrm{side}}\)，并由 \(m^*/m_0=7.619964/|C_{\mathrm{side}}|\) 得到图像估算的方向相关有效质量。同时，为降低图像分辨率和近似线性化带来的误差，程序保留左右两侧平均绝对斜率与最大绝对斜率作为辅助色散指标，并通过拟合点数、曲率符号、拟合 RMSE 和二次项贡献对曲率可靠性进行标记。该方法用于高通量预筛选阶段识别低有效质量、高迁移率潜力材料，最终有效质量和迁移率仍需基于原始能带数据或输运计算进一步验证。
"""
    out_path.write_text(text, encoding="utf-8")



def _prompt_float(prompt: str, default: float, min_value: Optional[float] = None) -> float:
    """Prompt for one floating-point value with validation."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("  输入无效：请输入数字，例如 0.10")
            continue
        if min_value is not None and value < min_value:
            print(f"  输入无效：数值不能小于 {min_value}")
            continue
        return value


def _prompt_float_range(prompt: str, default_min: float, default_max: float, min_value: Optional[float] = None) -> Tuple[float, float]:
    """Prompt for two floating-point values separated by spaces or comma."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return float(default_min), float(default_max)
        raw = raw.replace(",", " ").replace("，", " ")
        parts = [p for p in raw.split() if p]
        if len(parts) != 2:
            print("  输入无效：请一次输入两个数字，例如 0.10 0.50")
            continue
        try:
            vmin, vmax = float(parts[0]), float(parts[1])
        except ValueError:
            print("  输入无效：请一次输入两个数字，例如 0.10 0.50")
            continue
        if min_value is not None and (vmin < min_value or vmax < min_value):
            print(f"  输入无效：数值不能小于 {min_value}")
            continue
        if vmin > vmax:
            print("  输入无效：下限不能大于上限")
            continue
        return vmin, vmax



def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no input with validation."""
    default_txt = "y" if default else "n"
    while True:
        raw = input(prompt).strip().lower()
        if raw == "":
            return bool(default)
        if raw in ["y", "yes", "1", "true", "t"]:
            return True
        if raw in ["n", "no", "0", "false", "f"]:
            return False
        print(f"  输入无效：请输入 y 或 n，直接回车采用默认值 {default_txt}")


def _prompt_curvature_range(
    prompt: str,
    default_min: float,
    default_max: float,
) -> Tuple[float, float]:
    """
    Prompt for an absolute-curvature range.

    Accepted examples:
        15 30
        25 inf
        0 15
    """
    def _fmt(v: float) -> str:
        return "inf" if not np.isfinite(v) else f"{v:.3f}"

    while True:
        raw = input(prompt).strip().lower()
        if raw == "":
            return float(default_min), float(default_max)

        raw = raw.replace(",", " ").replace("，", " ")
        parts = [p for p in raw.split() if p]
        if len(parts) != 2:
            print("  输入无效：请输入两个值，例如 15 30 或 25 inf")
            continue

        try:
            vmin = float(parts[0])
            if parts[1] in {"inf", "+inf", "infinity", "+infinity"}:
                vmax = float("inf")
            else:
                vmax = float(parts[1])
        except ValueError:
            print("  输入无效：请输入两个值，例如 15 30 或 25 inf")
            continue

        if vmin < 0 or vmax < 0:
            print("  输入无效：绝对曲率范围不能小于 0")
            continue
        if vmin > vmax:
            print("  输入无效：曲率下限不能大于上限")
            continue
        return vmin, vmax


def interactive_parameter_setup(args: argparse.Namespace) -> argparse.Namespace:
    """
    VASPKIT-like interactive setup.

    Input files are auto-detected from ./input:
        *.dat, *.hdf5, *.h5

    Interaction order:
    1. band-gap screening range;
    2. whether direct-gap screening is required;
    3. curvature fitting energy window;
    4. absolute-curvature screening range.

    Only after all parameters are confirmed does main() read ./input and
    generate standardized images in ./band_images.
    """
    if bool(getattr(args, "no_interactive", False)):
        return args
    if not sys.stdin.isatty():
        return args

    print("\n")
    print("=" * 76)
    print(" Band-edge Curvature Screening: Interactive Setup")
    print("=" * 76)
    print("直接回车表示采用方括号中的默认值。")
    print(f"\n原始输入文件夹: {args.raw_input_dir}")
    print("程序将自动识别其中的 .dat、.hdf5 和 .h5 文件，无需选择输入格式。")

    print("\n[1] 带隙筛选范围 Eg_min Eg_max (eV)")
    args.gap_min_eV, args.gap_max_eV = _prompt_float_range(
        f"    请输入带隙范围 [默认: {args.gap_min_eV:.3f} {args.gap_max_eV:.3f}]: ",
        args.gap_min_eV,
        args.gap_max_eV,
        min_value=0.0,
    )

    print("\n[2] 直接带隙筛选")
    print("    若启用，则最终候选还要求 CBM 与 VBM 位于同一 k 点附近。")
    args.direct_gap_only = _prompt_yes_no(
        "    是否只筛选直接带隙材料? [默认: y]: ",
        default=True,
    )
    if args.direct_gap_only:
        args.direct_gap_tolerance_norm = _prompt_float(
            f"    请输入 Δk_direct（归一化 k 坐标）[默认: {args.direct_gap_tolerance_norm:.4f}]: ",
            args.direct_gap_tolerance_norm,
            min_value=0.0,
        )
    else:
        print(
            f"    不将直接带隙作为最终候选必要条件；"
            f"仍按 Δk={args.direct_gap_tolerance_norm:.4f} 统计直接带隙。"
        )

    print("\n[3] 曲率拟合能量范围 ΔE_fit (eV)")
    print("    CBM: E <= E_CBM + ΔE_fit")
    print("    VBM: E >= E_VBM - ΔE_fit")
    args.energy_window = _prompt_float(
        f"    请输入 ΔE_fit [默认: {args.energy_window:.3f}]: ",
        args.energy_window,
        min_value=0.001,
    )

    print("\n[4] 曲率筛选范围 |κ| (eV·Å²)")
    print("    对 CBM/VBM 使用曲率绝对值筛选，同时保留原有曲率符号物理检查。")
    print("    示例：15 30 表示 15 <= |κ| <= 30；25 inf 表示 |κ| >= 25。")
    default_max_txt = (
        "inf" if not np.isfinite(args.curvature_max_eVA2)
        else f"{args.curvature_max_eVA2:.3f}"
    )
    args.curvature_min_eVA2, args.curvature_max_eVA2 = _prompt_curvature_range(
        f"    请输入 |κ| 范围 [默认: {args.curvature_min_eVA2:.3f} {default_max_txt}]: ",
        args.curvature_min_eVA2,
        args.curvature_max_eVA2,
    )

    max_txt = (
        "inf" if not np.isfinite(args.curvature_max_eVA2)
        else f"{args.curvature_max_eVA2:.4f}"
    )

    print("\n当前参数设置：")
    print(f"    原始输入文件夹: {args.raw_input_dir}")
    print("    自动识别格式: DAT + HDF5/H5")
    print(f"    标准能带图文件夹: {args.input}")
    print(f"    带隙范围: {args.gap_min_eV:.4f} – {args.gap_max_eV:.4f} eV")
    print(f"    是否要求直接带隙: {'是' if args.direct_gap_only else '否'}")
    print(f"    直接带隙容差 Δk_direct: {args.direct_gap_tolerance_norm:.4f}")
    print(f"    曲率拟合能量窗口 ΔE_fit: {args.energy_window:.4f} eV")
    print(
        f"    曲率筛选范围: {args.curvature_min_eVA2:.4f} <= |κ| <= {max_txt} eV·Å²"
    )
    print("=" * 76)
    print("\n参数设置完成，开始自动读取 DAT/HDF5 并生成统一格式 band_images ...\n")
    return args


def _to_bool_series(s: pd.Series) -> pd.Series:
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

    # v8：优先使用 process_one 里按分段映射算出的 Å^-1 距离，
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
        gap_selected = _to_bool_series(df["gap_range_selected"])
    else:
        gap_selected = pd.Series(False, index=df.index)
    df["gap_and_direct_gap_selected"] = gap_selected & direct_selected

    df["direct_gap_condition_passed"] = True
    if bool(args.direct_gap_only):
        df["direct_gap_condition_passed"] = direct_selected
        if "final_selected" in df.columns:
            old_final = _to_bool_series(df["final_selected"])
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

def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Band curvature and slope extraction from standardized band images.")
    parser.add_argument(
        "--input",
        default=str(script_dir / "band_images"),
        help="Generated standardized band-image folder. Default: ./band_images",
    )
    parser.add_argument(
        "--raw_input_dir",
        default=str(script_dir / "input"),
        help="Folder containing raw DAT and/or HDF5/H5 files. Default: ./input",
    )
    parser.add_argument(
        "--hdf5_bandstructure",
        default=None,
        help="Optional HDF5 Bandstructure_N group. Default: latest group in each file",
    )
    parser.add_argument(
        "--hdf5_subtract_fermi",
        action="store_true",
        help="Subtract fermi_level when converting HDF5. Default: OFF",
    )
    parser.add_argument(
        "--standard_plot_dpi",
        type=int,
        default=300,
        help="DPI for automatically generated standardized images. Default: 300",
    )
    parser.add_argument(
        "--standard_plot_line_width",
        type=float,
        default=1.0,
        help="Band-line width for automatically generated images. Default: 1.0",
    )
    parser.add_argument("--output_dir", default=str(script_dir / "band_curvature_out"), help="Output folder. Default: ./band_curvature_out")
    parser.add_argument("--emin", type=float, default=-1.0, help="Energy minimum in images, eV")
    parser.add_argument("--emax", type=float, default=1.0, help="Energy maximum in images, eV")
    parser.add_argument("--fermi", type=float, default=0.0, help="Reference energy, usually 0 eV")
    parser.add_argument("--crop", default="auto", help="auto or left,top,right,bottom")
    parser.add_argument("--mode", choices=["blue_dark", "blue", "dark"], default="blue_dark", help="Band-line detection mode")
    parser.add_argument("--dark_threshold", type=int, default=90, help="Gray threshold for dark pixels")
    parser.add_argument("--margin", type=float, default=0.02, help="Energy margin around 0 eV, eV")
    parser.add_argument("--metal_gap_threshold_eV", type=float, default=0.08, help="Treat image-estimated gap <= this value as metallic/semimetallic and skip CBM/VBM curvature fitting, eV")
    parser.add_argument("--fermi_crossing_tol_eV", type=float, default=0.025, help="Energy tolerance around Fermi level for detecting band crossing, eV")
    parser.add_argument("--min_fermi_crossing_columns", type=int, default=8, help="Minimum number of k columns with band pixels near Fermi level for metallic detection")
    parser.add_argument("--min_fermi_crossing_fraction", type=float, default=0.005, help="Minimum fraction of k columns with band pixels near Fermi level for metallic detection")
    parser.add_argument("--local_width", type=float, default=0.10, help="Half width of local fitting window in normalized k")
    parser.add_argument("--energy_window", type=float, default=0.10, help="Energy window near CBM/VBM for curvature fitting, eV. Default: 0.10 to reduce branch mixing")
    parser.add_argument("--min_side_points", type=int, default=3, help="Minimum points for each left/right side fit")
    parser.add_argument("--rmse_threshold_eV", type=float, default=0.08, help="RMSE threshold for reliable curvature")
    parser.add_argument("--min_quadratic_span_eV", type=float, default=0.015, help="Minimum quadratic contribution energy to avoid near-linear flag")
    parser.add_argument("--linear_preference_tol", type=float, default=0.10, help="If linear RMSE <= quadratic RMSE*(1+tol), mark near-linear")
    parser.add_argument("--gap_min_eV", type=float, default=0.1, help="Minimum image-estimated band gap for semiconductor screening, eV")
    parser.add_argument("--gap_max_eV", type=float, default=0.5, help="Maximum image-estimated band gap for semiconductor screening, eV")
    parser.add_argument("--direct_gap_only", action="store_true", help="Require direct band gap for final_selected. In interactive mode this option is asked explicitly.")
    parser.add_argument("--direct_gap_tolerance_norm", type=float, default=0.01, help="Tolerance for direct-gap judgment in normalized k-path coordinate, default 0.01")
    parser.add_argument(
        "--curvature_min_eVA2",
        type=float,
        default=25.0,
        help="Minimum absolute curvature for curvature screening, eV*A^2. Default: 25",
    )
    parser.add_argument(
        "--curvature_max_eVA2",
        type=float,
        default=float("inf"),
        help="Maximum absolute curvature for curvature screening, eV*A^2. Default: inf",
    )
    parser.add_argument("--linear_mean_slope_threshold_eVA", type=float, default=3.0, help="Near-linear high-dispersion threshold using mean absolute slope, eV*A")
    parser.add_argument("--linear_max_slope_threshold_eVA", type=float, default=5.0, help="Near-linear high-dispersion threshold using max absolute slope, eV*A")
    parser.add_argument("--ktotal", type=float, default=None, help="Use one total k-path length for all images, unit A^-1")
    parser.add_argument("--path_length_csv", default=str(script_dir / "path_lengths.csv"), help="CSV with material_id,path_length_Ainv")
    parser.add_argument("--tesseract_cmd", default=None, help="Explicit tesseract executable path, e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    parser.add_argument("--save_ocr_regions", action="store_true", help="Save OCR crop regions for debugging Path length recognition")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug images. Default: debug images are generated.")
    parser.add_argument("--no_interactive", action="store_true", help="Disable VASPKIT-like interactive prompts and use command-line/default parameters directly.")
    args = parser.parse_args()
    args = interactive_parameter_setup(args)

    # Automatically convert all DAT/HDF5/H5 files -> standardized PNG images.
    generated_path_lengths, segments_map, preprocessing_summary = prepare_standard_band_images(args)

    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)

    # v8：第三方图像的分段信息通道。用户可在 band_images 下提供
    # kpath_segments_external.json（格式与自动生成的 kpath_segments.json
    # 相同：{图片名: {labels, boundaries_Ainv, source}}），由 kpath_segments.py
    # 从结构文件/化学式算出段边界后手工或脚本写入。该文件不会被 prepare
    # 阶段覆盖或删除；同名材料以本轮自动生成的信息优先。
    external_json = input_dir / "kpath_segments_external.json"
    if external_json.exists():
        try:
            ext = json.loads(external_json.read_text(encoding="utf-8"))
            n_added = 0
            for k, v in ext.items():
                key = safe_stem(Path(str(k)).stem)
                if key not in segments_map:
                    segments_map[key] = v
                    n_added += 1
            print(f"External k-path segments: {external_json} (+{n_added} materials)")
        except Exception as exc:
            print(f"WARNING: cannot parse {external_json}: {exc}")

    # 每次运行都先彻底清空上一轮 band_curvature_out，
    # 包括旧 CSV、debug_images、selected_debug_images、ocr_regions 等。
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug_images"
    ocr_region_dir = output_dir / "ocr_regions" if args.save_ocr_regions else None
    if not args.no_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if ocr_region_dir is not None:
        ocr_region_dir.mkdir(parents=True, exist_ok=True)

    path_csv = Path(args.path_length_csv) if args.path_length_csv else None
    path_lengths = load_path_lengths(path_csv)
    # v8：不再把渲染阶段的全局路径长度并入筛选端（v7 的内存旁路已移除）。
    # 生成的材料统一走 segments_map 分段标定；第三方图优先通过
    # kpath_segments_external.json 提供分段信息，外部 CSV / OCR / --ktotal
    # 仅作为完全没有分段信息时的全局标定回退链。

    images = list_images(input_dir)
    if not images:
        raise SystemExit(f"No images found in {input_dir}. Put images into ./band_images first.")

    rows = []
    for i, img_path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            rows.append(process_one(img_path, args, path_lengths, segments_map, debug_dir, ocr_region_dir))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows.append({
                "material_id": guess_material_id(img_path),
                "image_file": str(img_path),
                "debug_image": "",
                "path_length_Ainv": "",
                "path_length_source": "error",
                "path_length_ocr_text": "",
                "image_gap_eV": "",
                "processing_status": "error",
                "screen_out_reason": str(exc),
                "manual_notes": str(exc),
            })

    df = pd.DataFrame(rows)
    if "material_id" in df.columns:
        df = df.sort_values("material_id", ascending=True)

    df = apply_direct_gap_screening(df, args)

    # Summary counts after band-gap and direct-gap recognition.
    if "gap_range_selected" in df.columns:
        gap_hit_count = int(_to_bool_series(df["gap_range_selected"]).sum())
    else:
        gap_hit_count = 0

    if "direct_gap_selected" in df.columns:
        direct_hit_count = int(_to_bool_series(df["direct_gap_selected"]).sum())
    else:
        direct_hit_count = 0

    if "gap_and_direct_gap_selected" in df.columns:
        gap_direct_hit_count = int(_to_bool_series(df["gap_and_direct_gap_selected"]).sum())
    else:
        gap_direct_hit_count = 0

    if "curvature_selected" in df.columns:
        curvature_range_hit_count = int(_to_bool_series(df["curvature_selected"]).sum())
    else:
        curvature_range_hit_count = 0

    full_en = output_dir / "band_curvature_slope_results_english.csv"
    full_cn = output_dir / "band_curvature_slope_results.csv"

    # Compact final result files:
    # keep only the four direction-resolved curvature values and convert all
    # of them to absolute values. material_id is kept as the CSV row index so
    # each material can still be identified without adding another data column.
    result_curvature_cols = [
        "cbm_left_curvature_eVA2",
        "cbm_right_curvature_eVA2",
        "vbm_left_curvature_eVA2",
        "vbm_right_curvature_eVA2",
    ]

    compact_results = pd.DataFrame(index=df.index)
    for c in result_curvature_cols:
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

    compact_results_cn = compact_results.rename(columns={
        "cbm_left_curvature_eVA2": "CBM左侧绝对曲率_eV_A2",
        "cbm_right_curvature_eVA2": "CBM右侧绝对曲率_eV_A2",
        "vbm_left_curvature_eVA2": "VBM左侧绝对曲率_eV_A2",
        "vbm_right_curvature_eVA2": "VBM右侧绝对曲率_eV_A2",
    })
    compact_results_cn.index.name = "材料名称"
    compact_results_cn.to_csv(full_cn, index=True, encoding="utf-8-sig")

    # v8：另存全量明细，便于审计逐段标定质量（k_calibration_mode、
    # 分隔线检测偏差等诊断列都在其中）。
    detail_en = output_dir / "band_curvature_full_details_english.csv"
    detail_cn = output_dir / "band_curvature_full_details.csv"
    df.to_csv(detail_en, index=False, encoding="utf-8-sig")
    to_chinese_columns(df).to_csv(detail_cn, index=False, encoding="utf-8-sig")

    selected_counts = copy_selected_debug_images(df, output_dir) if not args.no_debug else {}

    debug_count = len(list(debug_dir.glob("*_debug.png"))) if debug_dir.exists() else 0
    print("\nDone.")
    if preprocessing_summary:
        print("\nInput preprocessing summary:")
        for source_name, stat in preprocessing_summary.items():
            print(
                f"  {source_name}: found={stat.get('found', 0)}, "
                f"success={stat.get('success', 0)}, failed={stat.get('failed', 0)}"
            )
        print(f"  Standardized images: {input_dir}")

    print(f"Curvature results CN: {full_cn}")
    print(f"Curvature results EN: {full_en}")
    print(f"Debug images: {debug_dir}  count={debug_count}")

    print("\nScreening summary:")
    print(f"  带隙范围 {args.gap_min_eV:.4f}–{args.gap_max_eV:.4f} eV 命中数量: {gap_hit_count}")
    print(f"  识别为直接带隙的数量: {direct_hit_count}")
    print(f"  带隙范围 + 直接带隙同时满足的数量: {gap_direct_hit_count}")
    curvature_max_txt = (
        "inf" if not np.isfinite(args.curvature_max_eVA2)
        else f"{args.curvature_max_eVA2:.4f}"
    )
    print(
        f"  曲率范围 {args.curvature_min_eVA2:.4f}–{curvature_max_txt} eV·Å² 命中数量: "
        f"{curvature_range_hit_count}"
    )

    if selected_counts:
        selected_root = output_dir / "selected_debug_images"
        print(f"Selected debug images: {selected_root}")
        for k, v in selected_counts.items():
            print(f"  {k}: {v}")
    if "path_length_source" in df.columns:
        print("\nPath length sources:")
        print(df["path_length_source"].value_counts(dropna=False).to_string())
    if "k_calibration_mode" in df.columns:
        print("\nK-calibration modes:")
        print(df["k_calibration_mode"].value_counts(dropna=False).to_string())
    print(f"Full detail CSV: {detail_en}")


if __name__ == "__main__":
    main()
