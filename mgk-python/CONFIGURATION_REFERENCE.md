# MGK1D 使用与参数参考

本文档对应 MGK1D 0.1.2 的公开 Python API。完整配置是一个包含
`physical`、`geometry`、`model`、`species`、`grid` 和 `solver` 的嵌套
Python mapping；只需提供需要修改的字段，其余由程序补齐。未记录的
字段会立即触发 `ValueError`，因此拼写错误不会被静默忽略。

近期六后端性能图和 KBM 精度图使用的完整数值参数、扫描顺序、分支锚点、
缓存开关及计时统计口径见 [BENCHMARK_CASES.md](BENCHMARK_CASES.md)。GPU 安装、
精度和性能调优见 [GPU_GUIDE.md](GPU_GUIDE.md)。

## 1. 最小运行方法

```python
import mgk

config = {
    "geometry": {"model": "s-alpha", "q": 1.0, "magneticShear": 1.0},
    "model": {"electronClosure": "adiabatic", "parallelBoundary": "periodic"},
    "grid": {"numTheta": 65, "numEnergy": 16, "numPitch": 24,
             "numBouncePoints": 32},
    "solver": {"useGpu": False, "blockPrecision": "double",
               "eigenTolerance": 1e-8},
}

result = mgk.solve(config)
print("omega R/v_ti =", result.omega)
print("omega [rad/s] =", result.omegaPhysical)
print("frequency [Hz] =", result.frequencyHz)
print("growth rate [1/s] =", result.growthRate)
print("eigen residual =", result.eigenResidual)
print("field residual =", result.fieldConstraintResidual)
```

`result.omega` 采用 `omega*R/v_ti` 归一化，其中
`v_ti=sqrt(T_i/m_i)`。模态约定为 `exp(-i*omega*t)`，因而
`Im(omega)>0` 表示增长。正式结果应使用 double precision，并同时检查
`eigenResidual` 和 `fieldConstraintResidual`。

## 2. `physical`: 物理量

本组除温度使用 eV 外均使用 SI 单位。

| 字段 | 有效默认值 | 单位 | 含义/约束 |
| --- | ---: | --- | --- |
| `magneticField` | `2.0` | T | 磁场强度，必须为正 |
| `majorRadius` | `1.7` | m | 主半径 `R`，必须为正 |
| `minorRadius` | `0.0` | m | 小半径 `a`，要求 `0 <= a < R`；Miller 几何要求 `a>0` |
| `ionMass` | `2*amu` | kg | 主离子质量 |
| `ionChargeNumber` | `1` | `e` | 主离子电荷数，不能为零 |
| `ionTemperature` | `1000.0` | eV | 主离子温度 |
| `electronTemperature` | `1000.0` | eV | 电子温度 |
| `electronTemperatureGradientLength` | `ionTemperatureGradientLength` | m | 电子温度梯度标长 `LTe` |
| `electronBeta` | `0.0` | 1 | `beta_e`；启用任一电磁场时必须大于零 |
| `densityGradientLength` | `1.7/4` | m | 密度梯度标长 `Ln` |
| `ionTemperatureGradientLength` | `1.7/10` | m | 离子温度梯度标长 `LTi` |
| `binormalWavenumber` | 对应 `ky*rho_i=0.45/sqrt(2)` | 1/m | 次法向波数 `ky`，显式给定时必须为正 |

`electronTemperatureGradientLength=None` 和 `binormalWavenumber=None` 是代码中的原始占位值；
上表写的是 `solve()` 真正使用的有效默认值。

## 3. `geometry`: 磁几何

| 字段 | 默认值 | 含义/约束 |
| --- | ---: | --- |
| `model` | `"s-alpha"` | `"s-alpha"`、`"miller"` 或 `"stellarator"` |
| `q` | `1.0` | 安全因子，不能为零 |
| `magneticShear` | `1.0` | 磁剪切 `s_hat` |
| `alpha` | `0.0` | s-alpha/Miller 压强梯度参数 |
| `ballooningAngle` | `0.0` | ballooning angle `theta0`，弧度 |
| `shiftDerivative` | `0.0` | Miller `Delta'` |
| `elongation` | `1.0` | Miller elongation `kappa`，必须大于零 |
| `elongationShear` | `0.0` | Miller `s_kappa` |
| `triangularity` | `0.0` | Miller triangularity `delta`，要求 `-1<delta<1` |
| `triangularityShear` | `0.0` | Miller `s_delta` |
| `betaStar` | `0.0` | Miller `beta_star` |
| `tableResolution` | `1001` | 几何表分辨率，必须是不小于 5 的奇数 |
| `profileFile` | `None` | stellarator `.npz` profile 路径；stellarator 时必填 |
| `fieldLineLabel` | `0.0` | stellarator 场线标签 |
| `mirrorConvention` | `None` | 兼容旧输入；新配置建议写到 `model.mirrorConvention` |

非周期 stellarator profile 必须使用 open boundary，且 profile 必须覆盖整个
`[thetaMin, thetaMax]`。如果 profile 含有 `vmecQ` 元数据且未显式设置 `q`，程序会使用
`vmecQ`。

## 4. `model`: 物理闭合与边界

| 字段 | 默认值 | 含义/约束 |
| --- | ---: | --- |
| `magneticMirror` | `True` | 保留的兼容开关；当前公开求解器始终使用 orbit 离散 |
| `aparallel` | `False` | 加入 `A_parallel`，得到 `[phi,A_parallel]` |
| `bparallel` | `False` | 加入 `B_parallel`；通常与 `aparallel=True` 一起使用 |
| `parallelBoundary` | `"periodic"` | `"periodic"`、`"open"`、`"open-extrapolated"` 或 `"open-dtn"` |
| `boundaryCoordinateStretch` | `1.0` | 坐标拉伸，必须 `>=1`，非 1 时只能用 `open-extrapolated` |
| `boundarySpongeStrength` | `0.0` | sponge 强度，必须非负 |
| `boundarySpongeFraction` | `0.30` | sponge 占比，要求 `0<value<=0.5` |
| `boundaryTailPeriods` | `1` | `open-dtn` 外延周期数，非负整数 |
| `boundaryTailPoints` | `17` | `open-dtn` 每周期点数，至少 4 |
| `mirrorConvention` | `"cgyro_s_alpha"` | `"physical"` 或 `"cgyro_s_alpha"` |
| `electronClosure` | `"auto"` | `"auto"`、`"adiabatic"` 或 `"kinetic"`；`"massless"` 会被拒绝 |

若未显式指定 `mirrorConvention`，Miller、stellarator 或任一电磁配置会自动使用
`"physical"`，其他 s-alpha 静电配置使用 `"cgyro_s_alpha"`。`open-dtn` 不允许坐标
拉伸、sponge 或 `thetaMapAlpha!=0`。

电磁两场和三场的核心配置为：

```python
two_field = {
    "physical": {"electronBeta": 0.02},
    "model": {"aparallel": True, "bparallel": False},
}
three_field = {
    "physical": {"electronBeta": 0.02},
    "model": {"aparallel": True, "bparallel": True},
}
```

## 5. `species`: 电子组分

| 字段 | 默认值 | 含义/约束 |
| --- | ---: | --- |
| `enabled` | `False` | 是否接受 `items` 中的显式电子配置 |
| `items` | `[]` | 一个 electron mapping 或只含一个 electron mapping 的 list |

当 `items` 非空时必须设置 `enabled=True`。当前主离子由 `physical` 定义，
`items` 只允许覆盖一个电子组分的以下字段：

| electron 字段 | 默认值 | 单位/含义 |
| --- | ---: | --- |
| `name` | `"electron"` | 名称 |
| `kind` | `"electron"` | 必须为 `"electron"` |
| `kinetic` | `False` | 是否作为动理学粒子处理 |
| `mass` | 电子质量 | kg |
| `chargeNumber` | `-1` | 电荷数，必须为负 |
| `temperature` | `physical.electronTemperature` | eV |
| `densityFraction` | `abs(ionChargeNumber)` | 相对主离子密度，必须满足电中性 |
| `densityGradientLength` | `physical.densityGradientLength` | m |
| `temperatureGradientLength` | `physical.electronTemperatureGradientLength` | m |

最小动理学电子输入为：

```python
"model": {"electronClosure": "kinetic", "parallelBoundary": "open"},
"species": {
    "enabled": True,
    "items": {"kind": "electron", "kinetic": True},
},
```

`electronClosure="auto"` 会根据 electron 的 `kinetic` 开关选择绝热或动理学闭合。
显式选择 `"kinetic"` 时 electron 必须 `kinetic=True`；显式选择
`"adiabatic"` 时必须为 `False`。

## 6. `grid`: 离散网格

| 字段 | 有效默认值 | 含义/约束 |
| --- | ---: | --- |
| `thetaMin` | `-4*pi` | 场线坐标下限，弧度 |
| `thetaMax` | `4*pi` | 场线坐标上限，必须大于 `thetaMin` |
| `numTheta` | `129` | theta 点数，至少 3 |
| `thetaMapAlpha` | `0.0` | theta 映射聚集参数，必须非负 |
| `energyMax` | `12.5` | 归一化能量截断，必须为正 |
| `numEnergy` | `28` | 能量点数，至少 2 |
| `numPitch` | `32` | pitch 点数，至少 2 |
| `numBouncePoints` | `32` | 每个 trapped orbit 的 bounce 积分点数，至少 8 |

注意：代码底层的原始占位值为 `numEnergy=16`、`numPitch=16`，但当用户省略
这两项时，配置归一化会将有效默认值设为上表的 28 和 32。benchmark
使用的网格是为各个算例显式设定的，不应与默认网格混淆。

## 7. `solver`: 后端、精度与特征求解

| 字段 | 有效默认值 | 含义/约束 |
| --- | ---: | --- |
| `useGpu` | 自动探测 | `False` 选 CPU，`True` 选 CUDA；可复现脚本应显式设置 |
| `blockPrecision` | `"single"` | `"single"` 或 `"double"`；正式结果应显式用 double |
| `eigenBackend` | 随设备选择 | CPU 为 `"eigs"`，GPU 为 `"gpu_arnoldi"` |
| `modeSelection` | `"nearest"` | `"nearest"` 或 `"max_growth_scan"` |
| `singleShiftTimeLimit` | `5.0` | s，单个 shift 时间限制，允许正无穷 |
| `eigenTolerance` | `1e-7` | 特征求解容差，必须为正 |
| `eigenSubspaceDimension` | `8` | Arnoldi/子空间维数，至少 2 |
| `eigenHotSubspaceDimension` | `5` | warm Ritz 热路径子空间，至少 3 |
| `enableWarmRitz` | `True` | 扫描中是否启用 warm Ritz |
| `eigenMaxIterations` | `None` | 最大迭代数；`None` 使用后端默认 |
| `eigenInitialVector` | `None` | continuation 使用的上一点 reduced mode |
| `gpuArnoldiMaxRestarts` | `3` | GPU Arnoldi 最大 restart 次数，至少 1 |
| `cpuFactorizationWorkers` | 自动，最多 `2*CPU` 且上限 16 | `0` 表示自动；否则是非负整数 |
| `cpuOperatorWorkers` | 自动，最多 8 | `0` 表示自动；否则是非负整数 |
| `frequencyGuess` | 内置 ITG guess | 物理角频率复数 shift，单位 rad/s；程序内部再除以 `v_ti/R` |
| `checkBlockFactorization` | `False` | 是否做 block factorization 诊断检查 |
| `enableFactorizationCache` | 跟随 GPU cache 开关 | 通用 factorization cache 开关 |
| `enableGpuFactorizationCache` | `True` | GPU factorization/response cache |
| `enableGpuResultCache` | `True` | 完整输入相同时直接复用上次结果 |
| `compactResult` | `False` | 是否返回精简的内部结果 |
| `returnMatrices` | `False` | 保留的旧接口；设为 `True` 会被 orbit-only 求解器拒绝 |

`frequencyGuess` 是有量纲的 rad/s，而 `result.omega` 和 `result.normalizedConfig.eigsGuess`
是 `R/v_ti` 归一化值。参数扫描时可将上一点的 `result.omegaPhysical` 作为下一点
`frequencyGuess`，并将 `result.reducedMode` 作为 `eigenInitialVector`。

## 8. 使用建议与常见错误

- 论文、对外报告和 CPU/GPU 精度对比：显式设置 `useGpu`、
  `blockPrecision="double"`、网格和容差，同时保存两个 residual。
- GPU single 只用于预扫描；最终点必须用 double checkpoint 复核。
- 动理学电子通常使用 `parallelBoundary="open"`，并需单独跟踪 ITG/TEM 分支。
- 电磁计算必须使用正 `electronBeta`；两场设 `aparallel=True`，三场再设
  `bparallel=True`。
- 不要只看代数 residual 就判定结果收敛；还要扫描 theta domain、`numTheta`、
  `numEnergy`、`numPitch`、`numBouncePoints` 和 `energyMax`。
- 性能比较必须固定网格、物理模型、shift 策略、continuation、精度、缓存开关和
  cold-anchor 排除规则；详细口径见 benchmark 文档。
