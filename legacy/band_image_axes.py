#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
band_image_axes.py

从任意文献风格能带图中识别坐标轴信息的模块（"任意图判断"第 2 步）：

    1. calibrate_y_axis   y 轴能量刻度 OCR + 等差先验 RANSAC 修复
                          （负号/小数点丢失自动纠正）→ 像素行 ↔ eV 线性标定
    2. read_x_labels      x 轴高对称点标签识别：逐 blob 单字符 OCR
                          （eng + 希腊语 ell 双语投票）+ 同形字映射 +
                          有限候选集吸附（Γ 由 ell 模型或混淆表识别）
    3. detect_x_anchors   高对称点像素位置：图框内竖直分隔线 ∪ 轴下刻度线，
                          再与标签质心配对（多余锚点自动丢弃，可抗竖直
                          能带的假阳性）
    4. analyze_axes       整图编排：图框检测 → y 标定 → 标签 + 锚点配对

依赖：tesseract（brew 安装）+ 同目录 tessdata/（eng + ell traineddata）。
所有阈值都在 100-300 dpi 的合成文献图上实测过（见会话实验记录）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_TESSDATA = Path(__file__).resolve().parent / "tessdata"

GREEK2LAT = {
    "Χ": "X", "Μ": "M", "Κ": "K", "Τ": "T", "Ζ": "Z", "Α": "A", "Η": "H",
    "Ν": "N", "Ρ": "P", "Υ": "Y", "Ε": "E", "Β": "B", "Ι": "I", "Ο": "O",
}
CANDSET = {
    "Γ", "X", "M", "K", "L", "W", "Z", "R", "A", "H", "N", "P", "S", "T", "U",
    "Y", "Σ", "Λ", "Δ", "F", "B", "C", "D", "E", "G", "V",
}
# eng 模型对 Γ 的常见误读（大小写敏感：小写 'r' 是 Γ，真实标签 R 读作 'R'；
# 数字/标点不可能是合法标签，'7'/';' 是斜体 Γ 的实测误读字形）
CONFUSION = {
    "r": "Γ", "I": "Γ", "|": "Γ", "l": "Γ", "[": "Γ", "(": "Γ", "{": "Γ",
    "7": "Γ", ";": "Γ", "1": "Γ",
}


@dataclass
class YCalibration:
    """像素行 -> 能量的线性标定：E = slope * row + intercept（row 为整图坐标）。"""
    slope: float
    intercept: float
    n_ticks: int
    n_inliers: int
    n_repaired: int
    max_resid_eV: float
    tick_rows: List[int] = field(default_factory=list)
    tick_values: List[float] = field(default_factory=list)

    def energy_at(self, row: float) -> float:
        return self.slope * float(row) + self.intercept


@dataclass
class AxesResult:
    frame: Tuple[int, int, int, int]          # L, R, T, B（spine 中心像素）
    ycal: Optional[YCalibration]
    labels: List[str]                          # 归一化后的高对称点标签序列
    label_x_px: List[float]                    # 与 labels 对齐的锚点像素 x
    label_q: List[float]                       # 锚点在图框内的归一化位置
    anchor_sources: List[str]                  # vline / tick / label_centroid
    warnings: List[str] = field(default_factory=list)


def find_frame(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """检测坐标轴图框（近满长的暗直线）。返回 (L, R, T, B) 或 None。"""
    binv = (gray < 128).astype(np.uint8)
    h, w = binv.shape
    xs = np.where(binv.sum(axis=0) > 0.55 * h)[0]
    ys = np.where(binv.sum(axis=1) > 0.55 * w)[0]
    if xs.size < 2 or ys.size < 2:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


# ---------------------------------------------------------------- y axis --


def _repair_fit(pairs: List[Tuple[int, float]], tol: float = 0.05):
    """等差先验 RANSAC：候选修正 {v, -v, v/10, -v/10}（丢负号/丢小数点）。

    返回 (score, slope, intercept, fixed_pairs)；fixed_pairs 中无法归线的
    点为 (row, None)。评分优先内点数，其次未修改值数量。
    """
    best = None
    for i, j in combinations(range(len(pairs)), 2):
        (r1, v1), (r2, v2) = pairs[i], pairs[j]
        if r1 == r2:
            continue
        for s1 in (v1, -v1, v1 / 10, -v1 / 10):
            for s2 in (v2, -v2, v2 / 10, -v2 / 10):
                a = (s2 - s1) / (r2 - r1)
                # 物理先验：能量随像素行号增大而减小，slope 必为负。
                # 否则全负刻度整体丢负号时，恒等假设会以更高"未修改数"
                # 得分胜出，把轴确定性翻转。
                if a >= 0:
                    continue
                b = s1 - a * r1
                fixed, n_in = [], 0
                for (r, v) in pairs:
                    pred = a * r + b
                    cands = (v, -v, v / 10, -v / 10)
                    d = [abs(pred - c) for c in cands]
                    k = int(np.argmin(d))
                    if d[k] < tol * max(1.0, abs(pred)) + 0.02:
                        n_in += 1
                        fixed.append((r, cands[k]))
                    else:
                        fixed.append((r, None))
                n_orig = sum(1 for (r, v), (_, f) in zip(pairs, fixed) if f == v)
                score = (n_in, n_orig)
                if best is None or score > best[0]:
                    best = (score, a, b, fixed)
    return best


def calibrate_y_axis(gray: np.ndarray, frame: Tuple[int, int, int, int]) -> Tuple[Optional[YCalibration], List[str]]:
    """OCR y 轴数字刻度并线性标定。失败返回 (None, warnings)。"""
    import pytesseract

    warnings: List[str] = []
    L, R, T, B = frame
    h, w = gray.shape

    # 1) 轴线左侧窄带内的向外刻度线 -> 行锚点
    x0 = max(0, L - 6)
    band = gray[:, x0:max(x0 + 1, L - 1)] < 128
    rows = np.where(band.sum(axis=1) >= 3)[0]
    ticks: List[int] = []
    if rows.size:
        splits = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
        ticks = [int(np.mean(s)) for s in splits if len(s)]
    tickless = not ticks

    # 2) 刻度文本区：轴线左侧、排除最左边的旋转轴标题
    strip = gray[:, : max(1, L - 8)]
    binv = (strip < 128).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(binv, connectivity=8)
    comps = [
        (stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], cent[i])
        for i in range(1, n)
        if 2 <= stats[i, 3] <= 0.08 * h and stats[i, 2] <= 0.08 * w
        and cent[i][0] > 0.30 * L
    ]
    if tickless:
        # 无向外刻度的图：把文本组件按行聚类，行质心作为锚点
        ys_sorted = sorted(c[4][1] for c in comps)
        ticks = []
        for y in ys_sorted:
            if not ticks or y - ticks[-1] > 0.02 * h + 4:
                ticks.append(int(y))
        warnings.append("y_axis_no_outward_ticks_using_text_lines")

    row_tol = max(8.0, 0.015 * h)
    pairs: List[Tuple[int, float]] = []
    n_rejected = 0
    for ty in ticks:
        grp = [c for c in comps if abs(c[4][1] - ty) < row_tol]
        if not grp:
            continue
        gx0 = min(c[0] for c in grp)
        gx1 = max(c[0] + c[2] for c in grp)
        gy0 = min(c[1] for c in grp)
        gy1 = max(c[1] + c[3] for c in grp)
        pad = 3
        crop = strip[max(gy0 - pad, 0): gy1 + pad, max(gx0 - pad, 0): gx1 + pad]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        txt = pytesseract.image_to_string(
            crop, config="--psm 7 -c tessedit_char_whitelist=0123456789.-"
        ).strip()
        txt = txt.replace("−", "-").replace("—", "-").replace("–", "-")
        if re.fullmatch(r"-?\d+(\.\d+)?", txt):
            pairs.append((ty, float(txt)))
        else:
            n_rejected += 1

    if len(pairs) < 3:
        warnings.append(f"y_axis_ocr_insufficient_ticks:{len(pairs)}")
        return None, warnings

    best = _repair_fit(pairs)
    if best is None:
        warnings.append("y_axis_repair_failed")
        return None, warnings
    (n_in, n_orig), a0, b0, fixed = best
    inliers = [(r, v) for (r, v) in fixed if v is not None]
    if len(inliers) < 3:
        warnings.append(f"y_axis_ransac_inliers_lt3:{len(inliers)}")
        return None, warnings

    rr = np.array([p[0] for p in inliers], dtype=float)
    vv = np.array([p[1] for p in inliers], dtype=float)
    slope, intercept = np.polyfit(rr, vv, 1)
    resid = np.abs(np.polyval([slope, intercept], rr) - vv)
    n_repaired = sum(
        1 for (r, v), (_, f) in zip(pairs, fixed) if f is not None and f != v
    )
    if n_repaired:
        warnings.append(f"y_axis_values_repaired:{n_repaired}")
    if float(resid.max()) > 0.05 * max(1.0, float(np.abs(vv).max())):
        warnings.append(f"y_axis_fit_resid_high:{float(resid.max()):.4f}")
    ycal = YCalibration(
        slope=float(slope),
        intercept=float(intercept),
        n_ticks=len(pairs),
        n_inliers=len(inliers),
        n_repaired=n_repaired,
        max_resid_eV=float(resid.max()),
        tick_rows=[int(r) for r in rr],
        tick_values=[float(v) for v in vv],
    )
    return ycal, warnings


# ---------------------------------------------------------------- x axis --


def _split_subscript(crop: np.ndarray) -> Tuple[np.ndarray, bool]:
    """把 K₁ 这类带下标的 blob 拆出主字形。"""
    binv = (crop < 160).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    comps = sorted([stats[i] for i in range(1, n) if stats[i, 4] >= 2], key=lambda s: s[0])
    if len(comps) >= 2:
        h_main = max(c[3] for c in comps)
        mains = [c for c in comps if c[3] > 0.75 * h_main]
        subs = [c for c in comps if c[3] <= 0.75 * h_main and c[1] > crop.shape[0] * 0.3]
        if subs and mains:
            x0 = min(c[0] for c in mains)
            x1 = max(c[0] + c[2] for c in mains)
            y0 = min(c[1] for c in mains)
            y1 = max(c[1] + c[3] for c in mains)
            return crop[y0:y1, x0:x1], True
    return crop, False


def _ocr_one_label(crop: np.ndarray, tessdata_dir: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """单个标签 blob 的识别：eng 与 eng+ell 双语投票 + 三级吸附。"""
    import pytesseract

    crop, _has_sub = _split_subscript(crop)
    up = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    up = cv2.copyMakeBorder(up, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
    votes: List[Tuple[str, str]] = []
    for lang in ["eng", "eng+ell"]:
        try:
            t = pytesseract.image_to_string(
                up, lang=lang, config=f'--tessdata-dir "{tessdata_dir}" --psm 10'
            ).strip()
        except Exception:
            continue
        if t:
            votes.append((lang, t))
    for _lang, t in votes:                       # 1) 精确命中 / 希腊同形字
        c = GREEK2LAT.get(t, t)
        if c in CANDSET:
            return c, votes
    for _lang, t in votes:                       # 2) 大小写敏感混淆表（须在大写化前）
        if t in CONFUSION:
            return CONFUSION[t], votes
    for _lang, t in votes:                       # 3) 大写化 + 多字符取主字形
        s = t.strip()
        if not s:
            continue
        c0 = GREEK2LAT.get(s[0].upper(), s[0].upper())
        if c0 in CANDSET:
            return c0, votes
    return (votes[0][1] if votes else ""), votes


def read_x_labels(
    gray: np.ndarray,
    frame: Tuple[int, int, int, int],
    tessdata_dir: Path = DEFAULT_TESSDATA,
) -> List[Tuple[float, str]]:
    """识别底轴下方的高对称点标签，返回 [(质心像素x, 标签)]（按 x 排序）。"""
    L, R, T, B = frame
    strip = gray[B + 3:, :]
    if strip.size == 0:
        return []
    binv = (strip < 160).astype(np.uint8)
    # 只保留高对称点标签所在的行块，避免把 x 轴标题（如 "Wave
    # vector"）的单词误识别为标签（'Wave'->W、'vector'->V 都在候选集
    # 内）。先按组件清理贴着条带顶端的轴残留——刻度线残端（窄组件）
    # 与底轴 spine 的抗锯齿残尾（矮而宽）——再以 >=4 全空行分块、取
    # 第一个行块（即标签行）。
    n0, lab0, stats0, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    for i in range(1, n0):
        cxi = stats0[i, 0] + stats0[i, 2] / 2.0
        touches_top = stats0[i, 1] <= 1
        if cxi < L - 4 or cxi > R + 4:
            # 图框横向范围外的内容（如 y 轴底部刻度标签渗入条带）
            binv[lab0 == i] = 0
        elif touches_top and (stats0[i, 2] <= 4 or stats0[i, 3] <= 2):
            binv[lab0 == i] = 0
    row_has = binv.sum(axis=1) > 0
    nz = np.where(row_has)[0]
    if nz.size == 0:
        return []
    blk_start = int(nz[0])
    blk_end = strip.shape[0]
    empty_run = 0
    for r in range(blk_start, strip.shape[0]):
        if row_has[r]:
            empty_run = 0
        else:
            empty_run += 1
            if empty_run >= 4:
                blk_end = r - empty_run + 1
                break
    strip = strip[blk_start:blk_end, :]
    binv = binv[blk_start:blk_end, :]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    comps = sorted([stats[i] for i in range(1, n) if stats[i, 4] >= 3], key=lambda s: s[0])
    merged: List[List[int]] = []
    for s in comps:
        prev = merged[-1] if merged else None
        y_overlap = (
            prev is not None
            and min(prev[1] + prev[3], s[1] + s[3]) - max(prev[1], s[1]) > 0
        )
        if prev is not None and s[0] - (prev[0] + prev[2]) < 3 and y_overlap:
            m = merged[-1]
            x1 = max(m[0] + m[2], s[0] + s[2])
            y1 = max(m[1] + m[3], s[1] + s[3])
            nx, ny = min(m[0], s[0]), min(m[1], s[1])
            merged[-1] = [nx, ny, x1 - nx, y1 - ny, 0]
        else:
            merged.append(list(s[:5]))
    out: List[Tuple[float, str]] = []
    for m in merged:
        if m[2] < 2 or m[3] < 4:
            continue
        cx = m[0] + m[2] / 2.0
        # 只保留图框横向范围内的 blob，排除 y 轴刻度文本等左右两侧内容
        if cx < L - 4 or cx > R + 4:
            continue
        pad = 2
        y0 = max(0, m[1] - pad)
        x0 = max(0, m[0] - pad)
        crop = strip[y0: m[1] + m[3] + pad, x0: m[0] + m[2] + pad]
        label, _votes = _ocr_one_label(crop, tessdata_dir)
        if label:
            out.append((cx, label))
    return out


def detect_x_anchors(gray: np.ndarray, frame: Tuple[int, int, int, int]) -> Tuple[List[float], List[float]]:
    """高对称点候选锚点：(图框内竖直分隔线, 轴下刻度线) 的像素 x 列表。"""
    L, R, T, B = frame
    inner = gray[T + 2: B - 1, L + 2: R - 1]
    seps: List[float] = []
    if inner.size:
        darkish = (inner < 170).astype(np.uint8)
        # 垂直膨胀以桥接虚线/点线分隔线的间隙（点线列覆盖率可低至
        # ~0.25）；能带曲线经过一列只占少数行，膨胀后仍远低于阈值。
        darkish = cv2.dilate(darkish, np.ones((5, 1), np.uint8))
        cov = darkish.sum(axis=0) / max(1, darkish.shape[0])
        cols = np.where(cov > 0.60)[0]
        if cols.size:
            groups = np.split(cols, np.where(np.diff(cols) > 2)[0] + 1)
            seps = [float(np.mean(g)) + L + 2 for g in groups if len(g)]
    ticks: List[float] = []
    tickband = gray[B + 1: B + 5, L: R + 1] < 128
    if tickband.size:
        tcols = np.where(tickband.sum(axis=0) >= 2)[0]
        if tcols.size:
            tgroups = np.split(tcols, np.where(np.diff(tcols) > 2)[0] + 1)
            ticks = [float(np.mean(g)) + L for g in tgroups if len(g)]
    return seps, ticks


def analyze_axes(
    image_path: str,
    tessdata_dir: Path = DEFAULT_TESSDATA,
    labels_override: Optional[List[str]] = None,
) -> AxesResult:
    """整图编排：图框 → y 标定 → 标签识别 → 锚点配对。

    labels_override：OCR 失败/不可靠时手工给定标签序列（数量须与识别
    到的标签 blob 一致，位置仍来自图像）。
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)
    frame = find_frame(gray)
    if frame is None:
        raise RuntimeError(f"Cannot detect plot frame in {image_path}")
    L, R, T, B = frame
    warnings: List[str] = []

    ycal, ywarn = calibrate_y_axis(gray, frame)
    warnings.extend(ywarn)

    found = read_x_labels(gray, frame, tessdata_dir)

    seps, ticks = detect_x_anchors(gray, frame)
    anchors = sorted(seps + ticks)
    dedup: List[float] = []
    for a in anchors:
        if not dedup or a - dedup[-1] > 3:
            dedup.append(a)

    if labels_override is not None:
        labels = [str(l) for l in labels_override]
        if len(found) == len(labels):
            centroids = [x for x, _ in found]
        elif len(dedup) == len(labels):
            # OCR blob 数与给定标签数不符时退而用去重锚点位置按序配对；
            # 绝不把真实质心与合成位置混在同一列表里。
            warnings.append(
                f"labels_override_using_anchor_positions:blobs={len(found)}"
            )
            centroids = list(dedup)
        else:
            raise ValueError(
                f"labels_override count {len(labels)} matches neither OCR label blobs "
                f"({len(found)}) nor detected anchors ({len(dedup)}); cannot place labels reliably."
            )
    else:
        labels = [l for _, l in found]
        centroids = [x for x, _ in found]
        bad = [l for l in labels if l not in CANDSET]
        if bad:
            warnings.append(f"labels_not_in_candidate_set:{bad}")

    # 锚点配对：每个标签质心吸附到最近的 分隔线/刻度 锚点；多余锚点
    # （如竖直能带造成的分隔线假阳性）因无标签认领而被自然丢弃。
    tol = 0.03 * (R - L)
    label_x: List[float] = []
    sources: List[str] = []
    for cx in centroids:
        near = [a for a in dedup if abs(a - cx) <= tol]
        if near:
            a = min(near, key=lambda v: abs(v - cx))
            label_x.append(a)
            sources.append("vline" if a in seps else "tick")
        else:
            label_x.append(float(cx))
            sources.append("label_centroid")
    if any(s == "label_centroid" for s in sources):
        warnings.append("some_anchors_from_label_centroid_only")
    if sorted(label_x) != label_x:
        warnings.append("anchor_positions_not_monotonic")

    label_q = [(x - L) / max(1.0, float(R - L)) for x in label_x]
    return AxesResult(
        frame=frame,
        ycal=ycal,
        labels=labels,
        label_x_px=label_x,
        label_q=label_q,
        anchor_sources=sources,
        warnings=warnings,
    )


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        res = analyze_axes(p)
        print(f"== {p}")
        print(f"  frame L,R,T,B = {res.frame}")
        if res.ycal:
            e_top = res.ycal.energy_at(res.frame[2])
            e_bot = res.ycal.energy_at(res.frame[3])
            print(
                f"  y axis: E(top)={e_top:.3f} eV, E(bottom)={e_bot:.3f} eV "
                f"(ticks={res.ycal.n_ticks}, inliers={res.ycal.n_inliers}, "
                f"repaired={res.ycal.n_repaired}, max_resid={res.ycal.max_resid_eV:.4f})"
            )
        else:
            print("  y axis: CALIBRATION FAILED")
        for lbl, x, q, s in zip(res.labels, res.label_x_px, res.label_q, res.anchor_sources):
            print(f"  label {lbl!r:6} x={x:7.1f}  q={q:.4f}  ({s})")
        if res.warnings:
            print(f"  warnings: {res.warnings}")
