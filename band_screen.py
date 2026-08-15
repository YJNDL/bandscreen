#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
band_screen.py —— 批量能带图像筛选（bandscreen 包的命令行入口）。

流程：input/ 的 DAT/HDF5 → 标准化 band_images/*.png + 分段元数据 →
图像识别提取 CBM/VBM → 逐段标定的带边曲率/有效质量 → 带隙、直接带隙
与 |C| 范围筛选 → band_curvature_out/ 下的 CSV 与 debug 图。

直接运行（交互式设置参数，回车用默认值）：
    python band_screen.py
非交互（脚本/批处理）：
    python band_screen.py --no_interactive [其他参数]

第三方文献图：把 PNG 与 kpath_segments_external.json（用
analyze_band_image.py --emit_segments_json 生成）放进 band_images，
input 为空时会跳过渲染、直接筛选现有图片。

说明：本工具基于能带图片像素识别，输出用于高通量预筛的图像近似
描述符，不替代基于原始能带数据的严格有效质量/迁移率计算。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from bandscreen import (
    apply_direct_gap_screening,
    copy_selected_debug_images,
    guess_material_id,
    list_images,
    prepare_standard_band_images,
    process_one,
    safe_stem,
    to_bool_series,
    write_result_csvs,
)

SCRIPT_DIR = Path(__file__).resolve().parent


# ------------------------------ 交互式设置 ------------------------------


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


def _prompt_curvature_range(prompt: str, default_min: float, default_max: float) -> Tuple[float, float]:
    """绝对曲率范围输入，如 "15 30" 或 "25 inf"。"""
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
    """VASPKIT 风格交互设置：带隙范围 → 直接带隙 → 拟合能量窗口 → |κ| 范围。"""
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


# ------------------------------ 参数与主流程 ------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Band curvature and slope extraction from standardized band images.")
    # 目录与渲染
    p.add_argument("--input", default=str(SCRIPT_DIR / "band_images"), help="Generated standardized band-image folder. Default: ./band_images")
    p.add_argument("--raw_input_dir", default=str(SCRIPT_DIR / "input"), help="Folder containing raw DAT and/or HDF5/H5 files. Default: ./input")
    p.add_argument("--output_dir", default=str(SCRIPT_DIR / "band_curvature_out"), help="Output folder. Default: ./band_curvature_out")
    p.add_argument("--hdf5_bandstructure", default=None, help="Optional HDF5 Bandstructure_N group. Default: latest group in each file")
    p.add_argument("--hdf5_subtract_fermi", action="store_true", help="Subtract fermi_level when converting HDF5. Default: OFF")
    p.add_argument("--standard_plot_dpi", type=int, default=300, help="DPI for automatically generated standardized images. Default: 300")
    p.add_argument("--standard_plot_line_width", type=float, default=1.0, help="Band-line width for automatically generated images. Default: 1.0")
    # 图像格式约定
    p.add_argument("--emin", type=float, default=-1.0, help="Energy minimum in images, eV")
    p.add_argument("--emax", type=float, default=1.0, help="Energy maximum in images, eV")
    p.add_argument("--fermi", type=float, default=0.0, help="Reference energy, usually 0 eV")
    p.add_argument("--crop", default="auto", help="auto or left,top,right,bottom")
    p.add_argument("--mode", choices=["blue_dark", "blue", "dark"], default="blue_dark", help="Band-line detection mode")
    p.add_argument("--dark_threshold", type=int, default=90, help="Gray threshold for dark pixels")
    p.add_argument("--margin", type=float, default=0.02, help="Energy margin around 0 eV, eV")
    # 金属判定
    p.add_argument("--metal_gap_threshold_eV", type=float, default=0.08, help="Treat image-estimated gap <= this value as metallic/semimetallic and skip CBM/VBM curvature fitting, eV")
    p.add_argument("--fermi_crossing_tol_eV", type=float, default=0.025, help="Energy tolerance around Fermi level for detecting band crossing, eV")
    p.add_argument("--min_fermi_crossing_columns", type=int, default=8, help="Minimum number of k columns with band pixels near Fermi level for metallic detection")
    p.add_argument("--min_fermi_crossing_fraction", type=float, default=0.005, help="Minimum fraction of k columns with band pixels near Fermi level for metallic detection")
    # 拟合
    p.add_argument("--local_width", type=float, default=0.10, help="Half width of local fitting window in normalized k")
    p.add_argument("--energy_window", type=float, default=0.10, help="Energy window near CBM/VBM for curvature fitting, eV. Default: 0.10 to reduce branch mixing")
    p.add_argument("--min_side_points", type=int, default=3, help="Minimum points for each left/right side fit")
    p.add_argument("--rmse_threshold_eV", type=float, default=0.08, help="RMSE threshold for reliable curvature")
    p.add_argument("--min_quadratic_span_eV", type=float, default=0.015, help="Minimum quadratic contribution energy to avoid near-linear flag")
    p.add_argument("--linear_preference_tol", type=float, default=0.10, help="If linear RMSE <= quadratic RMSE*(1+tol), mark near-linear")
    # 筛选条件
    p.add_argument("--gap_min_eV", type=float, default=0.1, help="Minimum image-estimated band gap for semiconductor screening, eV")
    p.add_argument("--gap_max_eV", type=float, default=0.5, help="Maximum image-estimated band gap for semiconductor screening, eV")
    p.add_argument("--direct_gap_only", action="store_true", help="Require direct band gap for final_selected. In interactive mode this option is asked explicitly.")
    p.add_argument("--direct_gap_tolerance_norm", type=float, default=0.01, help="Tolerance for direct-gap judgment in normalized k-path coordinate, default 0.01")
    p.add_argument("--curvature_min_eVA2", type=float, default=25.0, help="Minimum absolute curvature for curvature screening, eV*A^2. Default: 25")
    p.add_argument("--curvature_max_eVA2", type=float, default=float("inf"), help="Maximum absolute curvature for curvature screening, eV*A^2. Default: inf")
    p.add_argument("--linear_mean_slope_threshold_eVA", type=float, default=3.0, help="Near-linear high-dispersion threshold using mean absolute slope, eV*A")
    p.add_argument("--linear_max_slope_threshold_eVA", type=float, default=5.0, help="Near-linear high-dispersion threshold using max absolute slope, eV*A")
    # 其他
    p.add_argument("--tesseract_cmd", default=None, help="Explicit tesseract executable path (only needed for OCR fallback on images without segment metadata)")
    p.add_argument("--save_ocr_regions", action="store_true", help="Save OCR crop regions for debugging Path length recognition")
    p.add_argument("--no_debug", action="store_true", help="Disable debug images. Default: debug images are generated.")
    p.add_argument("--no_interactive", action="store_true", help="Disable interactive prompts and use command-line/default parameters directly.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args = interactive_parameter_setup(args)

    # 1) 渲染：DAT/HDF5 -> 标准 PNG + 分段元数据（input 为空则跳过）。
    segments_map, preprocessing_summary = prepare_standard_band_images(args)

    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)

    # 2) 第三方图分段信息通道：kpath_segments_external.json
    #    （由 analyze_band_image.py --emit_segments_json 生成，或按
    #    {图片名: {labels, boundaries_Ainv, source}} 手工编写；prepare
    #    阶段不会覆盖/删除；同名材料以本轮自动生成的信息优先。）
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

    # 3) 清空上一轮输出目录，重建。
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug_images"
    ocr_region_dir = output_dir / "ocr_regions" if args.save_ocr_regions else None
    if not args.no_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if ocr_region_dir is not None:
        ocr_region_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        raise SystemExit(f"No images found in {input_dir}. Put images into ./band_images first.")

    # 4) 逐图处理。
    rows = []
    for i, img_path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            rows.append(process_one(img_path, args, segments_map, debug_dir, ocr_region_dir))
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

    # 5) 输出与统计。
    def _count(col: str) -> int:
        return int(to_bool_series(df[col]).sum()) if col in df.columns else 0

    gap_hit_count = _count("gap_range_selected")
    direct_hit_count = _count("direct_gap_selected")
    gap_direct_hit_count = _count("gap_and_direct_gap_selected")
    curvature_range_hit_count = _count("curvature_selected")

    paths = write_result_csvs(df, output_dir)
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

    print(f"Curvature results CN: {paths['compact_cn']}")
    print(f"Curvature results EN: {paths['compact_en']}")
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
        print(f"Selected debug images: {output_dir / 'selected_debug_images'}")
        for k, v in selected_counts.items():
            print(f"  {k}: {v}")
    if "path_length_source" in df.columns:
        print("\nPath length sources:")
        print(df["path_length_source"].value_counts(dropna=False).to_string())
    if "k_calibration_mode" in df.columns:
        print("\nK-calibration modes:")
        print(df["k_calibration_mode"].value_counts(dropna=False).to_string())
    print(f"Full detail CSV: {paths['detail_en']}")


if __name__ == "__main__":
    main()
