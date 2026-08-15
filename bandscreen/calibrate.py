# -*- coding: utf-8 -*-
"""横轴标定：归一化坐标 q <-> 物理波矢 k（Å^-1）。

主通道：结构侧段边界（kpath_segments*.json）+ 图像检测的分隔线位置
组合成分段线性映射（PiecewiseKmap）。回退通道：OCR 读取图上印的
"Path length" 总长做全局均匀映射。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


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
    4. 只有全局总长（OCR 回退）——均匀映射；
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


# ------------------- OCR 回退：图上印的 Path length 总长 -------------------


def parse_path_length_from_text(text: str) -> Optional[float]:
    """
    Robustly parse the numeric value after "Path length" from OCR text.

    OCR may read "Path length: 12.8166 Å^-1" as variants such as:
    "Path Iength: 12.8166 A^-1", "Pathlength 12.8166", or "Path length: 12,8166".
    只接受含 Path/length 上下文的匹配——无锚点兜底会把 y 轴刻度等
    无关数字静默当作路径总长（曲率按其平方错）。
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
