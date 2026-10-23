# 膨胀仪升温曲线复核：加权二次叠加相变台阶的反演

材料实验室用膨胀仪测样品的线膨胀。做法是从参考温度开始按固定步长升温，逐点记录温度与长度变化；仪器对每个点给出自己的不确定度（升温越高噪声越大）。控温段的中间会有一段恒温保持（温度重复），那段只在第一个点上带来新信息。

合金在升温途中会过一次相变：相变点上长度会多出一个台阶，台阶陡缓因炉次而异。上一班把复核程序 `src/expansion.py` 接上这批记录时，报告里只有一条二次曲线、没有任何不确定度，台阶整段被抹平成曲率；客户问这个膨胀系数能信到什么程度、相变点落在哪儿，都答不上来。

## 实验室那边

程序以 `src/expansion.py --data 台账 --out 证书` 被调用，这个形态不能改。

  ```
  python3 src/expansion.py --data 记录JSON路径 --out 报告JSON路径
  ```

- 记录结构：`reference_temperature` 是参考温度；`samples` 每点有 `t`（°C）、`L`（长度变化，µm）、`sigma`（该点不确定度，µm）；`inspection` 里是本次核查的判定规则（标称斜率、斜率容差百分比、残差上限、标称相变中点、相变中点容差）；`fit` 里有 `plateau_tolerance_c`（判定恒温保持的温度容差）、`min_used`（有效点下限）、`error_correlation`（相邻采样点误差的相关系数 ρ，缺省为 0）与 `common_mode_sigma`（所有点一起漂移的共模误差标准差 τ，缺省为 0）。

## 复核口径

修好 `src/expansion.py`：

1. 恒温保持段：温度与上一点相差小于 `plateau_tolerance_c` 的采样视为保持点，**每组连续保持点只保留第一个**，其余剔除、不参与拟合。
2. 模型：`L = a0 + a1·ΔT + a2·ΔT² + A·S((T − Tc)/w)`，其中 `ΔT = t - reference_temperature`，`S(z) = 1/(1 + exp(-z))` 是相变台阶，`T` 取采样温度的绝对值（°C）。`Tc` 是相变中点、`w` 是相变宽度、`A` 是台阶高度，三个都要从数据里估出来。
3. 参数定法：以白化后的残差平方和最小为准。给定 `Tc`、`w` 时 `[a0, a1, a2, A]` 是线性的（用加权线性最小二乘解），所以先在 `Tc`、`w` 上粗扫定位、再逐个细化。
4. 误差口径：`error_correlation` 与 `common_mode_sigma` 都为 0 时是**加权最小二乘**，权重取 `1/sigma²`（各点不确定度不同，不能当等权处理）；
   否则误差相关，协方差矩阵是 `Σ = D·R·D + τ²·J`，其中 `D = diag(sigma_i)`、`R_ij = ρ^|i-j|`、`J` 是全 1 矩阵、`τ` 取 `common_mode_sigma`；
   这时按广义最小二乘算：对 `Σ` 做 Cholesky 分解，每条设计列与观测先按 `D` 归一、再用因子前代白化，然后解同一个正规方程。
5. 参数不确定度：`sigma_hat² = 白化残差平方和 /(n - 6)`，参数协方差 = `sigma_hat²·(JᵀJ)⁻¹`，`J` 是白化空间的 6 列雅可比（常数项、`ΔT`、`ΔT²`、台阶、以及台阶对 `Tc` 与对 `w` 的偏导两列：`A·S(1-S)·(-1/w)` 与 `A·S(1-S)·(-(T-Tc)/w²)`），标准误取对角线开方。
6. 给出 `a1` 与 `a2` 的相关系数（由协方差矩阵对应元素归一化），以及 `chi2_red = 白化后残差平方和 /(n - 6)`；`error_correlation` 与 `common_mode_sigma` 原样回填到证书里。
7. 按记录里的 `inspection` 段做首件核查判定：
   - `slope_vs_nominal`：算 `a1` 相对 `slope_nominal_um_per_c` 的偏离百分比，超过 `slope_tolerance_pct` 就判不通过；
   - `residual_chi2`：`chi2_red` 超过 `max_residual_chi2` 就判不通过；
   - `midpoint_vs_nominal`：`Tc` 与 `midpoint_nominal_c` 的偏差（°C）超过 `midpoint_tolerance_c` 就判不通过；
   - `checks` 依次给出这三项，每项有 `name`、`value`（算出来的偏离百分比、`chi2_red` 或中点偏差）、`limit`（对应的容差或上限）、`passed`（这一项是否通过）；
   - `verdict`：三项都通过写 ACCEPT，有一项不通过写 REJECT。

## 证书格式

证书同样是一份 UTF-8 JSON，内容按下面的结构组织：

```json
{
  "recorded_points": 416,
  "fitted_points": 376,
  "degrees_of_freedom": 370,
  "reference_temperature": 25.0,
  "expansion_terms": {"intercept_um": -0.000477, "slope_um_per_c": 0.600018, "curvature_um_per_c2": 0.0008},
  "transition_terms": {"midpoint_c": 180.02, "width_c": 1.04, "amplitude_um": 2.199},
  "term_uncertainties": {"intercept_um": 0.000339, "slope_um_per_c": 1.5e-05, "curvature_um_per_c2": 4.0e-07},
  "transition_uncertainties": {"midpoint_c": 0.031, "width_c": 0.012, "amplitude_um": 0.0041},
  "slope_intercept_correlation": -0.968,
  "residual_chi2": 1.034,
  "verdict": "ACCEPT",
  "checks": [
    {"name": "slope_vs_nominal", "value": 0.03, "limit": 1.0, "passed": true},
    {"name": "residual_chi2", "value": 1.034, "limit": 1.5, "passed": true},
    {"name": "midpoint_vs_nominal", "value": 0.02, "limit": 3.0, "passed": true}
  ],
  "error_correlation": 0.0,
  "common_mode_sigma": 0.0,
  "certificate_note": {"method": "wls-transition", "unit": {"temperature": "degC", "length": "um"}}
}
```

证书要写清台账点数、去重后的拟合点数，自由度等于拟合点数减六。
- `expansion_terms`、`term_uncertainties` 与 `transition_terms` / `transition_uncertainties` 里的 `amplitude_um` 各保留**六位小数**；`transition_terms` / `transition_uncertainties` 里的 `midpoint_c` 与 `width_c` 保留三位。
- `slope_intercept_correlation` 与 `residual_chi2` 保留三位。
- `certificate_note.method` 写 `wls-transition`（等权不成立时按加权算）或 `gls-transition`（相关误差时），`certificate_note.unit` 照抄输入的单位段。
- 证书里的每个数都必须是有限值；`NaN` 与 `Infinity` 一旦出现，这份证书就不算数。

## 失败与重跑

合规返回 0；点数不足十二、`sigma` 非正、剔除后有效点少于 `min_used`、`fit.plateau_tolerance_c` 不是正数、或没有温度跨度时返回 2，并在 stderr 说清缘由，不生成证书。
同一份台账跑两次，证书要一模一样，别把生成时间写进去。
失败的这次运行不能改写上一张证书，旧文件保持原样。

## 改动范围

依赖只有解释器自带的模块，离线运行；允许改的文件是 `src/expansion.py`，别去碰参数名和输出字段。
- 证书里的每个数都只能由这次台账算出来，照抄样例不算；复检会换升温范围、膨胀系数与噪声水平。
