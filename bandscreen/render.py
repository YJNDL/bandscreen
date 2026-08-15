# -*- coding: utf-8 -*-
"""原始能带数据（DAT / QuantumATK HDF5）-> 标准化 PNG + 分段元数据。

渲染阶段是唯一允许接触原始数据的阶段；它把每个材料的高对称点标签
序列与各段边界累积 k 坐标写入 band_images/kpath_segments.json（声明
的晶体侧输入，不含绘图像素信息），供筛选阶段做逐段标定。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import HARTREE_TO_EV, IMAGE_EXTS, safe_stem


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
) -> Tuple[Dict[str, dict], Dict[str, int]]:
    """
    DAT format:
      first column = cumulative k-path coordinate (Å^-1)
      columns 2...N = band energies (eV)

    Optional KLABELS/POSCAR in the same directory are used when available.
    """
    files = find_dat_band_files(input_dir)
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

    return segments_map, stats


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
) -> Tuple[Dict[str, dict], Dict[str, int]]:
    files = find_hdf5_band_files(input_dir)
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

    return segments_map, stats


def prepare_standard_band_images(args) -> Tuple[Dict[str, dict], Dict[str, Dict[str, int]]]:
    """
    自动把 ./input 下的 DAT 与 HDF5/H5 能带数据转换为 ./band_images 的
    标准 PNG，并收集每材料的分段元数据写入 kpath_segments.json。

    - input 中没有原始文件时跳过（保留现有图片，支持"只筛第三方图"）；
    - 清理时只删上一轮自动生成的图片；kpath_segments_external.json
      声明的第三方图无条件保留。
    """
    raw_dir = Path(args.raw_input_dir)
    output_dir = Path(args.input)

    if not raw_dir.exists():
        raise SystemExit(f"Raw input folder not found: {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not find_dat_band_files(raw_dir) and not find_hdf5_band_files(raw_dir):
        print(
            f"WARNING: no .dat/.hdf5/.h5 files in {raw_dir}; "
            f"keeping existing images in {output_dir} and skipping regeneration."
        )
        return {}, {}

    # Rebuild band_images for the current batch only.
    old_json = output_dir / "kpath_segments.json"
    old_auto_stems: Optional[set] = None
    if old_json.exists():
        try:
            old_auto_stems = set(json.loads(old_json.read_text(encoding="utf-8")).keys())
        except Exception:
            old_auto_stems = None
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

    segments_all: Dict[str, dict] = {}
    summary: Dict[str, Dict[str, int]] = {}
    used_stems: set = set()

    dat_segments, dat_stats = convert_dat_band_files(
        input_dir=raw_dir,
        output_dir=output_dir,
        emin=args.emin,
        emax=args.emax,
        dpi=args.standard_plot_dpi,
        line_width=args.standard_plot_line_width,
        used_stems=used_stems,
    )
    segments_all.update(dat_segments)
    summary["DAT"] = dat_stats

    h5_segments, h5_stats = convert_hdf5_band_files(
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

    return segments_all, summary
