# MGK1D Python

MGK1D 是一个局域、无碰撞、线性陀螺动理学本征值求解器，用于计算等离子体中的 ITG、TEM 和电磁 KBM 模态。公开接口完全由 Python 提供，核心计算采用 passing/trapped-orbit 的 matrix-free 离散，可在 CPU 或 NVIDIA CUDA GPU 上运行。

## 主要功能

- s-alpha、Miller 和局域 stellarator 磁几何；
- 绝热电子和动理学电子；
- 静电单场 `[phi]`；
- 电磁两场 `[phi, A_parallel]`；
- 电磁三场 `[phi, A_parallel, B_parallel]`；
- CPU double、GPU double，以及用于预扫描的 GPU single precision；
- 本征值、场闭合残差、模态向量和计时信息；
- 论文算例的一键批量运行和结果输出。

当前版本不包含碰撞、旋转、全局径向物理、非线性演化或已弃用的旧网格后端。

## 安装

要求 Python 3.10--3.13、NumPy 2.x 和 SciPy 1.13 或更高版本。

CPU 环境：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

如需一次安装求解器、GPU、VMEC、测试和论文绘图所需的完整依赖，可使用根目录的 `requirements.txt`，然后安装本地包：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

仅运行 CPU 求解器时，可继续使用较小的 `requirements-cpu.txt`。

GPU 环境需要 64 位 Linux、兼容的 NVIDIA 驱动和对应的 CuPy 变体：

```bash
python -m pip install -e ".[gpu12]"   # CUDA 12
python -m pip install -e ".[gpu13]"   # CUDA 13
```

同一个环境中不要同时安装 `cupy-cuda12x` 和 `cupy-cuda13x`。Windows 可以尝试 CPU 后端，但当前版本未列入正式兼容性保证范围。

安装验证：

```bash
python -c "import mgk; print(mgk.__version__)"
python mgk/examples/cpu_quickstart.py
python -m pip install -e ".[test]"
python -m pytest -q
```

## 基本调用

稳定的公开入口是 `mgk.solve`、`mgk.Struct` 和 `mgk.__version__`。下面的配置计算 s-alpha 几何下的绝热电子 ITG：

```python
import math
import mgk

config = {
    "geometry": {
        "model": "s-alpha",
        "q": 1.0,
        "magneticShear": 1.0,
        "alpha": 0.0,
    },
    "model": {
        "electronClosure": "adiabatic",
        "parallelBoundary": "periodic",
    },
    "grid": {
        "thetaMin": -4 * math.pi,
        "thetaMax": 4 * math.pi,
        "numTheta": 65,
        "numEnergy": 16,
        "numPitch": 24,
        "numBouncePoints": 32,
    },
    "solver": {
        "useGpu": False,
        "eigenBackend": "eigs",
        "blockPrecision": "double",
        "eigenTolerance": 1e-8,
    },
}

result = mgk.solve(config)
print("omega R/v_ti =", result.omega)
print("omega [rad/s] =", result.omegaPhysical)
print("frequency [Hz] =", result.frequencyHz)
print("growth rate [1/s] =", result.growthRate)
print("eigen residual =", result.eigenResidual)
print("field residual =", result.fieldConstraintResidual)
```

模态时间约定为 `exp(-i*omega*t)`，因此 `Im(omega)>0` 表示不稳定增长。`result.omega` 以 `v_ti/R` 归一化，`result.omegaPhysical` 的单位为 rad/s。

### 动理学电子

动理学电子需要显式声明电子 species；通常使用开放平行边界：

```python
config["model"].update({
    "electronClosure": "kinetic",
    "parallelBoundary": "open",
})
config["species"] = {
    "enabled": True,
    "items": {"kind": "electron", "kinetic": True},
}
result = mgk.solve(config)
```

扫描 `k_y` 或其他参数时，应为 ITG/TEM 分支分别设置锚点，并将前一点的 `omegaPhysical` 和 `reducedMode` 传给下一点；不能只按每一点的最大增长率选择分支。

### 电磁模型

电磁计算必须设置正的 `physical.electronBeta`，并打开相应场：

```python
config["physical"] = {"electronBeta": 0.02}
config["model"].update({"aparallel": True, "bparallel": False})
result = mgk.solve(config)  # [phi, A_parallel]

config["model"]["bparallel"] = True
result = mgk.solve(config)  # [phi, A_parallel, B_parallel]
```

结果中的场数组分别为 `result.phi`、`result.aparallel` 和 `result.bparallel`，实际启用的场列表可通过 `result.fields` 查看。

生产计算应显式设置 `solver.useGpu` 和 `solver.blockPrecision`。论文或对外报告使用 double precision；GPU single 仅用于快速预扫描，并应在保留点用 double 复核。

## 论文七个 case

`paper_physics_cases/` 保存论文中的七个物理算例。每个目录包含一个 `run_case.py`、对应的 PDF 矢量图以及该算例所需的配置；批量入口为 `paper_physics_cases/run_all.py`。

| 编号 | 目录 | 内容 | 代表性输出 |
| --- | --- | --- | --- |
| 01 | `01_salpha_itg_eta` | s-alpha 几何、绝热电子 ITG 随 `eta_i` 扫描 | `salpha_itg_eta_cgyro_mgk.pdf` |
| 02 | `02_cbc_kinetic_itg_tem` | 动理学电子 CBC 中 ITG/TEM 分支和 Rewoldt 模态对比 | `cbc_kinetic_itg_tem_branches.pdf` |
| 03 | `03_strong_gradient_tem` | 强梯度动理学电子 TEM 模态比较 | `salpha_strong_gradient_tem_mode_cgyro_mgk.pdf` |
| 04 | `04_miller_triangularity_itg` | Miller 几何三角形变对绝热电子 ITG 的影响 | `miller_itg_triangularity_cgyro_mgk.pdf` |
| 05 | `05_miller_tem_domain` | 圆形/Miller 几何下 TEM 及平行计算域比较 | `miller_tem_domain_cgyro_mgk.pdf` |
| 06 | `06_xie_em_cbc` | Xie 等人的电磁 CBC/KBM 两场算例 | `xie2016_electromagnetic_cbc_benchmark.pdf` |
| 07 | `07_shen_em_kbm` | Shen 等人的大长宽比电磁 KBM 两场和三场算例 | `shen2025_electromagnetic_kbm_benchmark.pdf` |

运行单个 case（默认 CPU）：

```bash
python paper_physics_cases/01_salpha_itg_eta/run_case.py
```

使用 GPU 或运行完整扫描：

```bash
python paper_physics_cases/01_salpha_itg_eta/run_case.py --backend gpu
python paper_physics_cases/01_salpha_itg_eta/run_case.py --full
```

批量运行七个 case：

```bash
python paper_physics_cases/run_all.py --backend cpu
python paper_physics_cases/run_all.py --backend gpu --full
```

运行结果会写入相应 case 目录中的 `python_results.csv` 和 `python_validation.json`，批量状态写入 `paper_physics_cases/validation_summary.json`。PDF 图为随仓库提供的论文版输出。

## 仓库目录

```text
mgk/                    求解器公开包、内部数值模块、示例和测试
paper_physics_cases/    论文七个 case、运行脚本和 PDF 图
research_results/       精选验证结果、输入参数、CSV/JSON 数据和案例说明
figures/                已生成的比较图、诊断图和历史结果归档
tools/                  几何转换、stellarator 数据处理和发行辅助工具
mgk1d.py                旧脚本的兼容导入入口；新代码请使用 import mgk
pyproject.toml          包元数据、依赖和可选安装项
requirements-cpu.txt    CPU 环境依赖清单
```

`mgk.internal` 下的名称属于实现细节，版本升级时可能变化。配置字段、平台支持和 GPU 参数以当前源码和本 README 为准。

## 结果检查

正式结果至少记录 `omega`、`growthRate`、`fields`、`eigenResidual` 和 `fieldConstraintResidual`。建议检查：

```python
assert result.eigenResidual < 2e-6
assert result.fieldConstraintResidual < 2e-6
```

残差只说明离散后的本征问题已求解，并不替代数值收敛性检查。发表或对外报告前，还应改变 theta 域、`numTheta`、`numEnergy`、`numPitch`、`numBouncePoints` 和 `energyMax` 进行收敛测试。

## 许可证与支持范围

本仓库按项目发行包中的许可证和支持策略使用。当前正式支持的目标平台为 64 位 Linux/macOS CPU，以及带兼容 NVIDIA 驱动的 64 位 Linux GPU；其他平台可能可以运行，但未作同等保证。
