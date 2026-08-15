# -*- coding: utf-8 -*-
"""绘图区检测与能带像素提取：图像 -> (k_norm, E) 点集与 CBM/VBM 包络。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


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
    # 去噪用连通域面积过滤而非 2x2 形态学开运算：偶数核开运算在倾斜
    # 细线上有方向性偏移（腐蚀只保留与相邻列暗行重叠的部分：上升段
    # 整体偏下、下降段偏上），会造成带边两侧能量的系统性反向偏差，
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
