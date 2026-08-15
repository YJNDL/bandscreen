# 能带图像筛选工具（bandscreen）

从能带图像中提取带隙、直接/间接带隙、带边二阶曲率与图像估算有效质量，
用于高通量材料预筛选。输出是图像近似描述符，不替代基于原始能带数据的
严格有效质量/迁移率计算。

## 目录结构

```
8.12/
├── band_screen.py           批量筛选入口（交互式，替代旧 v7/v8 脚本）
├── analyze_band_image.py    任意单张文献图分析入口
├── bandscreen/              核心包（见各模块 docstring）
├── tessdata/                OCR 语言模型（eng + ell；Γ 识别必需）
├── input/                   原始能带数据（*.dat / *.hdf5 / *.h5）
├── band_images/             自动生成的标准化能带图 + 分段元数据
├── band_curvature_out/      筛选输出（CSV、debug 图、命中分类）
└── legacy/                  已被取代的历史单文件脚本（仅存档）
```

## 批量筛选（自己的计算数据）

```bash
python band_screen.py                # 交互式：带隙范围/直接带隙/ΔE_fit/|κ| 范围
python band_screen.py --no_interactive [--gap_min_eV 0.1 --gap_max_eV 0.5 ...]
```

流程：`input/` 的 DAT/HDF5 → 标准化 `band_images/*.png`（附
`kpath_segments.json` 分段元数据）→ 图像识别 CBM/VBM → **高对称路径
逐段标定**的固定顶点二次拟合（顶点吸附 / 顶点能量去偏 / 段边界截断）
→ 金属剔除、带隙/直接带隙/|C| 筛选 → `band_curvature_out/`：

- `band_curvature_slope_results(_english).csv` —— 精简表：四个方向的绝对曲率
- `band_curvature_full_details(_english).csv` —— 全量明细（含 k 标定模式、
  顶点吸附、分隔线检测偏差等诊断列）
- `debug_images/`、`selected_debug_images/` —— 叠加拟合曲线的核对图

## 任意文献图单图分析

```bash
python analyze_band_image.py --image fig3b.png --structure POSCAR --mode dark
python analyze_band_image.py --image fig.png --mp mp-149            # 需 MP_API_KEY
python analyze_band_image.py --image fig.png --structure a.cif \
    --labels "G,X,W,L,G"                                            # OCR 失败时手工给标签
```

y 轴能量与高对称点标签/位置全部从图像识别；晶体结构（POSCAR/CIF 或
Materials Project 化学式）是唯一的声明外部输入，用于计算各段物理长度。
输出 JSON + debug 叠加图；`--emit_segments_json band_images/kpath_segments_external.json`
可把该图接入批量管线（放同名 PNG 进 `band_images/`，`input/` 为空时
band_screen.py 会跳过渲染直接筛选现有图片）。

带边在图边缘时自动做自由顶点二次重拟合（JSON 中 `preferred` 字段给出
采信裁决）；y 轴 OCR 失败时用 `--emin/--emax` 手工给定能量窗口。

已知限制：多面板图（能带+DOS 并排）需先裁剪出能带面板。

## 依赖

Python 包：opencv-python、h5py、pandas、matplotlib、pytesseract、
pymatgen（仅任意图模式算段长）、mp-api（仅 `--mp` 从 Materials
Project 取结构时需要，另需 MP_API_KEY）。系统：tesseract（brew/apt
安装），希腊语模型已随 `tessdata/` 提供。

## 验证

- `python -m bandscreen.kpath --selftest` —— 段长计算对闭式解（fcc Γ-X=2π/a 等）
- 合成已知曲率的文献风格图（等比与各段等宽两种画法）端到端恢复 ±10% 内
- 37 材料批量输出与重构前版本逐列一致（回归基线）
