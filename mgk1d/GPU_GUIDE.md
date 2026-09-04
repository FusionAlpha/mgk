# MGK1D NVIDIA GPU 使用指南

本指南说明 MGK1D 的 CuPy/CUDA 后端如何安装、配置、验证和做性能测试。
GPU 后端使用矩阵无关的 passing/trapped-orbit 求解器，适合重复的参数扫描，
例如固定几何下扫描 `eta_i` 或 `k_y rho_i`。它不是独立的物理模型；GPU 与 CPU
后端应使用相同的输入、边界条件和离散网格进行物理结果比较。
所有通用配置字段的默认值、单位和约束见
[CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)。

## 1. 支持范围

GPU 后端的当前支持范围是：

- 64 位 Linux；
- NVIDIA GPU 和兼容的 NVIDIA 驱动；
- Python 3.10--3.13；
- CuPy CUDA 12（`cupy-cuda12x` 13.x--14.x），或 CuPy CUDA 13（14.x）；
- s-alpha、Miller 和预处理 stellarator field-line profile；
- electrostatic ITG/TEM，包含绝热或动理学电子；
- `periodic`、`open`、`open-extrapolated` 和 `open-dtn` 边界（具体物理组合仍受主文档限制）。

电静力和电磁多场 orbit 配置均使用 GPU 原生组装：passing/trapped 插值模板常驻
显存，与波数相关的 page、field response contraction、factorization 和 Arnoldi
均在 GPU 执行。多场路径支持 `[phi, A_parallel]` 和
`[phi, A_parallel, B_parallel]`，并在 GPU 侧组装相应的 gyro-average 因子及导数。
由于 RTX 5090/Blackwell 上的 Bessel kernel 兼容性，`J0/J1` 仍在 CPU 计算，但只
传输 Bessel 值，不再把完整 orbit page 拉回 CPU。可以用
`result.blockSolverInfo.gpuAssembly` 检查本次调用是否使用 GPU 原生组装。

Windows 和非 NVIDIA GPU 不在 0.1.2 的 GPU 认证范围内。GPU 依赖是可选依赖，
CPU 安装不需要 CUDA 或 CuPy。

## 2. 安装

先确认驱动能识别设备：

```bash
nvidia-smi
```

从源码 checkout 安装 CUDA 12 依赖：

```bash
python3 -m pip install -e '.[gpu12]'
```

CUDA 13 环境使用：

```bash
python3 -m pip install -e '.[gpu13]'
```

如果使用 wheel，则先安装 wheel，再安装与驱动匹配的 CuPy 变体：

```bash
python3 -m pip install mgk-0.1.2-py3-none-any.whl
python3 -m pip install 'cupy-cuda12x>=13,<15'
```

不要在同一个环境中同时安装 `cupy-cuda12x` 和 `cupy-cuda13x`。CuPy wheel
通常已经包含所需的 CUDA runtime Python 包；系统仍必须提供足够新的 NVIDIA
驱动。编译 CUDA kernel 不要求系统安装完整 CUDA toolkit，但某些自定义扩展
或集群环境可能仍需要管理员提供 toolkit。

安装后运行最小探测：

```bash
python3 - <<'PY'
import cupy as cp
import mgk

print("MGK1D", mgk.__version__)
print("CuPy", cp.__version__)
print("device_count", cp.cuda.runtime.getDeviceCount())
if cp.cuda.runtime.getDeviceCount():
    print("device", cp.cuda.Device().name)
PY
```

如果 `device_count` 为零，先检查作业是否申请了 GPU、
`CUDA_VISIBLE_DEVICES` 是否为空，以及驱动/runtime 的兼容性。

## 3. 明确选择 GPU 后端

生产脚本应显式设置 `solver.useGpu`，不要依赖自动探测。一个最小的动理学
电子 TEM 配置如下：

```python
import mgk

config = {
    "physical": {
        "magneticField": 2.0,
        "majorRadius": 1.0,
        "minorRadius": 0.18,
        "ionTemperature": 1000.0,
        "electronTemperature": 1000.0,
        "densityGradientLength": 1.0 / 2.22,
        "ionTemperatureGradientLength": 1.0 / 2.22,
        "electronTemperatureGradientLength": 1.0 / 6.92,
        "binormalWavenumber": 0.335,
    },
    "geometry": {"model": "s-alpha", "q": 1.4, "magneticShear": 0.776},
    "model": {
        "parallelBoundary": "open",
        "electronClosure": "kinetic",
    },
    "species": {
        "enabled": True,
        "items": {"kind": "electron", "kinetic": True},
    },
    "grid": {
        "numTheta": 129,
        "numEnergy": 24,
        "numPitch": 48,
        "numBouncePoints": 48,
    },
    "solver": {
        "useGpu": True,
        "eigenBackend": "gpu_arnoldi",
        "blockPrecision": "double",
        "eigenTolerance": 1e-8,
        "eigenSubspaceDimension": 8,
        "gpuArnoldiMaxRestarts": 12,
        "enableGpuFactorizationCache": True,
        "enableGpuResultCache": False,
    },
}

result = mgk.solve(config)
print("omega =", result.omega)
print("eigen residual =", result.eigenResidual)
print("field residual =", result.fieldConstraintResidual)
```

`eigenBackend="gpu_arnoldi"` 要求 `useGpu=True`。如果只设置
`useGpu=True` 而不设置 backend，当前配置归一化也会选择 GPU Arnoldi；为保证
脚本可读和结果可复现，仍建议两项都显式写出。

## 4. 精度选择

| 设置 | 用途 | 典型残差 | 建议 |
|---|---|---:|---|
| `blockPrecision="double"` | 论文、最终频率/增长率、交叉代码认证 | `eigenResidual` 通常约 `1e-12--1e-10` | 生产默认 |
| `blockPrecision="single"` | 快速扫描、参数预览、初始 mode 搜索 | 通常约 `1e-4` | 必须用 double checkpoint 复核 |

single precision 的频率通常仍能达到约 `1e-6` 的一致性，但完整特征残差可能为
`1e-4--1e-3`，不能按 double 的严格标准解释。最终报告应保存频率、
`eigenResidual` 和 `fieldConstraintResidual`，并用 double 复核同一物理分支。
double 生产结果可以使用类似检查：

```python
assert result.eigenResidual < 1e-7
assert result.fieldConstraintResidual < 1e-7
```

single 不应复用这个 double 阈值；应先确认频率相对 double 的误差满足任务要求，
再接受较宽的代数残差。这些检查只说明当前离散系统的代数求解可信，不等于
theta、energy、pitch、bounce 或能量截断已经收敛。发表结果仍需做离散分辨率
收敛测试。

## 5. Factorization cache 与 result cache

GPU 有两种不同的缓存：

- GPU assembly cache：自动缓存不随波数/温度梯度改变的 orbit topology 和
  interpolation template；兼容的温度梯度扫描只更新仿射系数；
- `enableGpuFactorizationCache=True`：缓存同一几何、shift、精度下的 orbit page
  分解和响应页。改变温度梯度等仿射参数时可以复用分解，这是重复扫描提速的
  关键设置；
- `enableGpuResultCache=True`：当完整输入完全相同时直接返回上一次结果，适合
  完全重复的调用，不适合需要新 eta 点的扫描。

缓存是进程内缓存，不跨 Python 进程、不跨 Slurm job。一个正常的 continuation
扫描通常表现为：

1. cold anchor 较慢，包含 CUDA context、CuPy JIT、首次 page factorization；
2. 第一个 changed point 可能负责建立 factorization cache，也较慢；
3. 后续 changed points 才代表稳定的热路径。

CuPy 编译出来的 kernel 缓存默认放在临时目录。集群上可以指定稳定位置：

```bash
export CUPY_CACHE_DIR="$SCRATCH/cupy-cache/mgk"
```

不要把 `.cupy_cache` 或用户目录绝对路径打进外部发布包。

## 6. 正确的性能测试方法

CPU/GPU 性能比较必须同时固定：

- theta、energy、pitch、bounce 网格；
- parallel boundary、geometry、species 和物理参数；
- Arnoldi 子空间和 tolerance；
- fixed shift 或 moving shift 策略；
- cold anchor 是否计入平均；
- single/double precision；
- 是否允许 factorization/continuation cache。

推荐把 anchor 排除，只报告 changed-point 的中位数，并单独报告第一个 changed
point。不要把 GPU single 与 CPU double 的残差或时间直接当作同一种算法比较。

一个可复现的 Slurm 作业应设置线程数并申请一张 GPU：

```bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
```

在固定 shift 的 eta 扫描中，当前实现的主要热路径通常是 GPU Arnoldi；在改变
`k_y` 的扫描中，GPU assembly 和 page factorization 也会重新执行。报告中应分别
保存 `result.timing.assembly`、`factorization` 和 `eigensolve`，不能只用总时间
猜测瓶颈。

## 7. 已验证性能参考

完整物理参数、分支跟踪方式、anchor 排除规则和精度验证见
[BENCHMARK_CASES.md](BENCHMARK_CASES.md)。本节只列性能摘要。

以下是 RTX 5090 Linux 节点 `liustation` 上 2026-08-30 的统一 kinetic CBC
orbit 记录，仅用于量级参考，不是跨硬件保证。配置为 open boundary、
`N_E=16, N_pitch=24, N_bounce=24`、`subspace=8`，29 个 ITG/TEM `k_y`
点中排除两个 anchor，报告其余 27 点的平均 wall time：

| `N_theta` | GPU double | GPU single |
|---:|---:|---:|
| 33 | `0.03287 s/点` | `0.03046 s/点` |
| 65 | `0.03971 s/点` | `0.03279 s/点` |
| 97 | `0.05315 s/点` | `0.03584 s/点` |
| 129 | `0.08027 s/点` | `0.04057 s/点` |

在 `N_theta=129`，优化前 Python GPU double/single 分别为 `0.52681` 和
`0.50002 s/点`，因此 GPU 原生组装带来 6.56x 和 12.32x 加速。同一硬件、
同一硬件上的历史参考值仅用于内部性能回归，不属于公开运行要求。
Python GPU double/single 当前分别快 2.57x 和 1.77x。完整原始 CSV、旧结果和
说明位于 `analysis/paper_electrostatic_performance/gpu_assembly_optimized/`。

## 8. 为什么 cold GPU 可能不如 CPU

这通常不是 CuPy 安装错误，而是测量口径或调用粒度问题：

- GPU 首次调用包含 CUDA context 初始化和 CuPy kernel 编译；
- 小矩阵 page factorization 的固定开销可能高于 CPU 的 MKL/LAPACK；
- host/device synchronization 会放大短任务的 wall time；
- 不同实现可能使用了已预热的固定 shift、内部 cache 和专用 Arnoldi 路径；
- single、double、不同子空间维数或不同 restart 数会改变迭代工作量。

旧 Python GPU 路径还曾在 CPU/NumPy 中完整组装 orbit page，再逐数组传到 GPU；
这会让 single 和 double 的 assembly 时间几乎相同。0.1.2 当前工作树已将电静力
和电磁多场组装迁移到 GPU。若新结果仍表现为 assembly 占总时间约 80%，先检查
运行的包路径是否确实指向包含 GPU 原生组装的版本，并打印
`result.blockSolverInfo.gpuAssembly`。

因此应分别报告 cold 和 warm，并优先优化重复扫描的稳定热路径。

## 9. 常见故障

### `No module named cupy`

当前环境没有安装 CUDA 变体。安装 `cupy-cuda12x` 或 `cupy-cuda13x`，不要
安装两个变体。

### `cudaErrorInsufficientDriver` / `cudaErrorInvalidDeviceFunction`

检查 `nvidia-smi` 的 driver 版本、CuPy CUDA 变体和节点实际 CUDA runtime。
容器或集群 module 环境中不要混用不匹配的 `LD_LIBRARY_PATH`。

### GPU 内存不足

先降低 `numEnergy`、`numPitch`、`numBouncePoints` 或 theta 范围，再逐步增加
分辨率。减少 Arnoldi 子空间只影响 eigensolver 工作量，不会线性减少所有 page
缓存的显存占用。一个 Python 进程内的 CuPy memory pool 也可能保留已释放块；
长脚本可在独立扫描阶段结束后调用：

```python
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
```

### 结果不稳定或频率跳支

先用 double、较大的 Arnoldi 子空间和足够的 restart 数确认 anchor，再按参数
连续方向使用前一点的 `eigenInitialVector`。高 `|theta|` 区域还必须检查 open
boundary、theta 域和 bounce 分辨率；GPU 加速不会修复离散或边界条件问题。

## 10. 复现清单

保存以下信息后，其他机器才能有意义地复现实验：

```text
MGK1D version
Python / NumPy / SciPy / CuPy versions
NVIDIA GPU name and driver version
CUDA runtime version
grid and boundary settings
blockPrecision and eigenBackend
eigen tolerance, subspace, restart limit
factorization/result cache settings
cold anchor and changed-point timing separately
eigenResidual and fieldConstraintResidual
```

GPU 后端只改变线性代数执行位置，不改变输入归一化和物理方程。任何 GPU/CPU
差异都应先通过 double-precision 频率、mode overlap、两个残差和离散收敛共同
判断，不能只看 wall time。
