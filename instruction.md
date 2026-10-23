# 膨胀仪升温曲线复核：加权二次拟合与参数不确定度

材料实验室用膨胀仪测样品的线膨胀。做法是从参考温度开始按固定步长升温，逐点记录温度与长度变化；仪器对每个点给出自己的不确定度（升温越高噪声越大）。控温段的中间会有一段恒温保持（温度重复），那段只在第一个点上带来新信息。

上一班把复核程序 `src/expansion.py` 接上这批记录时，报告里只有三个系数、没有任何不确定度，客户问这个膨胀系数能信到什么程度答不上来；换另一台升温更高、噪声更大的设备再跑，系数明显偏了。

## 现状

程序以 `src/expansion.py --data 台账 --out 证书` 被调用，这个形态不能改。

  ```
  python3 src/expansion.py --data 记录JSON路径 --out 报告JSON路径
  ```

- 记录结构：`reference_temperature` 是参考温度；`samples` 每点有 `t`（°C）、`L`（长度变化，µm）、`sigma`（该点不确定度，µm）；`fit` 里有 `plateau_tolerance_c`（判定恒温保持的温度容差）、`min_used`（有效点下限）与 `error_correlation`（相邻采样点误差的相关系数 ρ，缺省为 0）。

## 任务

修好 `src/expansion.py`：

1. 恒温保持段：温度与上一点相差小于 `plateau_tolerance_c` 的采样视为保持点，**每组连续保持点只保留第一个**，其余剔除、不参与拟合。
2. 模型：`L = a0 + a1·ΔT + a2·ΔT²`，`ΔT = t - reference_temperature`。
3. 拟合口径看 `error_correlation`：
   - 为 0 时是**加权最小二乘**，权重取 `1/sigma²`（各点不确定度不同，不能当等权处理）；
   - 不为 0 时误差相关，协方差矩阵是 `Σ = D·R·D`，其中 `D = diag(sigma_i)`、`R_ij = ρ^|i-j|`；
     这时是**广义最小二乘**：先把每条设计列与观测按 `D` 归一、再用 `R` 的 Cholesky 因子做前代白化，然后解同一个正规方程。
4. 给出参数的标准误：`sigma_hat² = 残差平方和 /(n - 3)`（这里残差是白化之后的残差），协方差矩阵 `sigma_hat²·(XᵀWX)⁻¹`，`W` 在相关时是 `Σ⁻¹`；标准误取对角线开方。
5. 给出 `a1` 与 `a2` 的相关系数（由协方差矩阵对应元素归一化）。
6. 给出 `chi2_red = 白化后残差平方和 /(n - 3)`，并把 `error_correlation` 原样回填到报告里。

## 输出契约

证书同样是一份 UTF-8 JSON，内容按下面的结构组织：

```json
{
  "recorded_points": 241,
  "fitted_points": 201,
  "degrees_of_freedom": 198,
  "reference_temperature": 25.0,
  "expansion_terms": {"intercept_um": -0.000477, "slope_um_per_c": 0.600018, "curvature_um_per_c2": 0.0008},
  "term_uncertainties": {"intercept_um": 0.000339, "slope_um_per_c": 1.5e-05, "curvature_um_per_c2": 4.0e-07},
  "slope_intercept_correlation": -0.968,
  "residual_chi2": 1.034,
  "error_correlation": 0.0,
  "certificate_note": {"method": "wls-quadratic", "unit": {"temperature": "degC", "length": "um"}}
}
```

证书要写清台账点数、去重后的拟合点数，自由度等于拟合点数减三。
- `coefficients` 与 `std_errors` 各保留**六位小数**；`corr_a1_a2` 与 `chi2_red` 保留三位。
- `meta.unit` 照抄输入；数值必须是有限值，不允许出现 `NaN` 或 `Infinity`。

## 运行与容错

合规返回 0；点数不足十二、`sigma` 非正、去重后不足 `min_used`、或没有温度跨度时返回 2，并在 stderr 说清缘由，不生成证书。
同一份台账跑两次，证书要一模一样，别把生成时间写进去。
失败的这次运行不能改写上一张证书，旧文件保持原样。

## 约束

依赖只有解释器自带的模块，离线运行；允许改的文件是 `src/expansion.py`，别去碰参数名和输出字段。
- 报告只能由当前记录算出来，不得写死样例数值；复检会喂别的升温范围、膨胀系数与噪声水平的记录。
