# -*- coding: utf-8 -*-
"""debug 叠加图绘制。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .fitting import EdgeResult


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

    # 图像检测到的高对称点分隔线位置（橙色），用于人工核对逐段标定。
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
