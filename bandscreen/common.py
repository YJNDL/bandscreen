# -*- coding: utf-8 -*-
"""公共常量与小工具。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np

IMAGE_EXTS = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
HBAR2_OVER_M0_EV_A2 = 7.619964  # ħ²/m0, unit: eV·Å²
HARTREE_TO_EV = 27.211386245988


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


def opt(v: Optional[float]) -> object:
    """CSV 友好化：None/NaN -> 空串，其余转 float。"""
    if v is None:
        return ""
    try:
        if not np.isfinite(v):
            return ""
    except Exception:
        return v
    return float(v)


def finite_float(v: Optional[float]) -> Optional[float]:
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
