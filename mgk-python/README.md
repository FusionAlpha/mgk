# MGK1D Python

项目仓库：[github.com/FusionAlpha/mgk](https://github.com/FusionAlpha/mgk)
许可证：BSD 3-Clause，版权归 FusionAlpha 所有。

MGK1D 是一个局域、无碰撞、线性陀螺动理学本征值求解器，主要用于
ITG、TEM 和电磁 KBM 研究。当前 Python 版本采用 matrix-free
passing/trapped-orbit 离散，支持：

- s-alpha 和 Miller 局域磁几何；
- 绝热电子或动理学电子；
- 静电 `[phi]`、电磁两场 `[phi,A_parallel]` 和三场
  `[phi,A_parallel,B_parallel]`；
- CPU 和 NVIDIA CUDA 后端；
- CPU/GPU double precision 正式计算与 GPU single precision 预扫描。

Python 是当前唯一的公开安装和使用方式。仓库和发行包不依赖商业软件。

## 1. Python 安装

### CPU 环境

要求 Python 3.10--3.13、NumPy 2.x 和 SciPy 1.13 或更高版本。从源码安装：

```bash
git clone https://github.com/FusionAlpha/mgk.git
cd mgk
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

从 wheel 安装：

```bash
python -m pip install ./mgk-0.1.2-py3-none-any.whl
```

### NVIDIA GPU 环境

需要 64 位 Linux、NVIDIA GPU 和兼容驱动。CUDA 12 环境安装：

```bash
python -m pip install -e '.[gpu12]'
```

CUDA 13 环境安装：

```bash
python -m pip install -e '.[gpu13]'
```

不要在同一 Python 环境中同时安装 `cupy-cuda12x` 和 `cupy-cuda13x`。
完整的驱动要求、缓存设置和性能调优见 [GPU_GUIDE.md](GPU_GUIDE.md)。

## 2. 安装验证

先检查导入和版本：

```bash
python -c "import mgk; print(mgk.__version__)"
```

运行 CPU smoke test：

```bash
python mgk/examples/cpu_quickstart.py
```

从源码树运行完整 Python 测试前，安装 test extra：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

GPU 机器可额外检查：

```bash
python - <<'PY'
import cupy as cp
print("CuPy", cp.__version__)
print("GPU count", cp.cuda.runtime.getDeviceCount())
if cp.cuda.runtime.getDeviceCount():
    print("GPU", cp.cuda.Device().name)
PY
```

## 3. 最小 Python 算例

以 CPU double precision 求解绝热电子 ITG：

```python
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
        "thetaMin": -4 * 3.141592653589793,
        "thetaMax": 4 * 3.141592653589793,
        "numTheta": 65,
        "numEnergy": 16,
        "numPitch": 24,
        "numBouncePoints": 32,
    },
    "solver": {
        "useGpu": False,
        "blockPrecision": "double",
        "eigenTolerance": 1e-8,
        "singleShiftTimeLimit": 60,
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

模态约定为 `exp(-i*omega*t)`，因此 `Im(omega)>0` 表示不稳定增长。
`result.omega` 以 `v_ti/R` 归一化，`result.omegaPhysical` 的单位是
rad/s。

## 4. 动理学电子

动理学电子需要显式启用 electron species，并建议使用 open parallel
boundary：

```python
config = {
    "geometry": {"model": "s-alpha", "q": 1.4, "magneticShear": 0.8},
    "model": {
        "electronClosure": "kinetic",
        "parallelBoundary": "open",
    },
    "species": {
        "enabled": True,
        "items": {"kind": "electron", "kinetic": True},
    },
    "grid": {
        "numTheta": 97,
        "numEnergy": 16,
        "numPitch": 32,
        "numBouncePoints": 48,
    },
    "solver": {
        "useGpu": False,
        "blockPrecision": "double",
        "eigenTolerance": 1e-8,
    },
}

result = mgk.solve(config)
```

ITG 和 TEM 可能同时存在。扫描 `k_y` 时应分别选择分支锚点，使用上一点的
`omegaPhysical` 和 `reducedMode` 做 continuation，不要仅按逐点最大增长率
判定同一分支。

## 5. 电磁两场和三场

电磁计算必须设置正的 `physical.electronBeta`。两场配置：

```python
config["physical"] = {"electronBeta": 0.02}
config["model"].update({"aparallel": True, "bparallel": False})
result = mgk.solve(config)
print(result.fields)  # ['phi', 'aparallel']
```

三场配置：

```python
config["model"].update({"aparallel": True, "bparallel": True})
result = mgk.solve(config)
print(result.fields)  # ['phi', 'aparallel', 'bparallel']
```

返回结果分别在 `result.phi`、`result.aparallel` 和 `result.bparallel`。

## 6. CPU/GPU 选择与精度

生产脚本应显式指定后端：

```python
# CPU double
config["solver"].update({
    "useGpu": False,
    "eigenBackend": "eigs",
    "blockPrecision": "double",
})

# GPU double
config["solver"].update({
    "useGpu": True,
    "eigenBackend": "gpu_arnoldi",
    "blockPrecision": "double",
    "enableGpuFactorizationCache": True,
    "enableGpuResultCache": False,
})
```

| 精度 | 用途 | 结果要求 |
| --- | --- | --- |
| `double` | 论文、报告、CPU/GPU 精度对比 | 同时检查 eigen 和 field residual |
| `single` | GPU 快速预扫描 | 必须在相同点用 double 复核 |

参数扫描做性能比较时，必须固定物理参数、网格、shift 策略、continuation、
精度、缓存开关和 cold-anchor 排除口径。

## 7. 结果验收

正式结果至少保存：

```python
print(result.omega)
print(result.eigenResidual)
print(result.fieldConstraintResidual)
print(result.fields)
print(result.timing)

assert result.eigenResidual < 2e-6
assert result.fieldConstraintResidual < 2e-6
```

代数 residual 合格不等于物理离散已收敛。发表或对外报告前还应扫描
theta domain、`numTheta`、`numEnergy`、`numPitch`、`numBouncePoints`
和 `energyMax`。

## 8. 参数与复现文档

- [research_results/case_studies/](research_results/case_studies/README.md)：四个典型算例的结果图、原始 CSV/JSON 数据、
  参数说明和复现脚本，是查看科研结果的首选入口。
- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)：全部 Python 配置字段、默认值、
  单位和约束。
- [GPU_GUIDE.md](GPU_GUIDE.md)：CUDA/CuPy 安装、精度、缓存、计时和故障排查。
- [BENCHMARK_CASES.md](BENCHMARK_CASES.md)：绝热电子、动理学电子、Shen 电磁
  和 Xie-2016 KBM 算例的数据来源与复现口径。
- [PUBLISHED_VALIDATION_CASES.md](PUBLISHED_VALIDATION_CASES.md)：论文算例、归一化、
  分支选择和数值 checkpoint。
- [COMPATIBILITY.md](COMPATIBILITY.md)：已测试的 Python/依赖/平台组合。

## 9. 项目结构

```text
mgk/                         Python 主程序、示例和测试（导入名保持为 mgk）
  examples/                    Python quick start 和验证算例配置
  tests/                       Python 回归测试
research_results/              精选算例和 benchmark 输入
figures/                       按算例分类的历史图片归档
analysis/                      本地原始扫描和中间结果，不进入安装包
tools/                         构建、转换和发布审计工具
```

主程序和典型算例的入口说明见 [mgk/README.md](mgk/README.md) 和
[research_results/README.md](research_results/README.md)。历史图片分类见
[figures/README.md](figures/README.md)。

Python 稳定公开入口是 `mgk.solve`、`mgk.Struct` 和 `mgk.__version__`。
`mgk.internal` 下的名称是实现细节，可能在后续版本变化。
旧脚本可以暂时使用 `import mgk1d` 兼容入口；新代码请统一使用 `import mgk`。

## 10. 支持边界

当前 Python 求解器不包含碰撞、旋转、全局径向物理、非线性演化、
massless-fluid electron closure 或旧 grid backend。许可证文本见
[LICENSE](LICENSE)。
