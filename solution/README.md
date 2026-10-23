# 难点与测试对应

## 难点 1：异方差必须加权，而且权重是 1/sigma²

推理轴：领域公式（加权最小二乘）

落在推理路径上的哪里：每个采样点自带不确定度 sigma，升温越高噪声越大。权重取 1/sigma² 才是最大似然；等权处理会把高温端的高噪声点和低温端的低噪声点同等对待，系数的标准误差出好几倍，`residual_chi2` 也从 1 附近跑到别的量级。原实现连 sigma 都没读。

对应测试：test_coefficients_match_independent_wls、test_std_errors_reflect_point_weights、test_chi2_reduced_near_unity、test_huge_sigma_point_barely_moves_fit

## 难点 2：相关误差与共模漂移要走广义最小二乘

推理轴：数值方法（Cholesky 白化）

落在推理路径上的哪里：`fit.error_correlation` 非零或 `fit.common_mode_sigma` 非零时，各点误差不再独立，协方差矩阵是 `Σ = D·R·D + τ²·J`。这时要整体做 Cholesky 分解，把设计列与观测一起前代白化，再解同一个正规方程；把相关误差当成独立误差算，斜率会偏、标准误明显偏小。原实现只按 1/sigma² 加权，共模项完全没进模型。

对应测试：test_correlated_errors_use_gls（门禁）、test_correlated_errors_std_error_matches_gls、test_common_mode_error_handled

## 难点 3：恒温保持段只留第一个点

推理轴：数据口径

落在推理路径上的哪里：控温平台有 40 个重复温度点，它们不带来新的温度信息，却把权重全堆在同一个温度上，等于人为放大该点的权重。规则是每组连续保持点只保留第一个，其余剔除。原实现把这些点全算进去了，拟合点数与系数都会偏。

对应测试：test_plateau_run_kept_once_only、test_second_dataset_with_other_range

## 难点 4：首件核查判定（业务规则）

推理轴：业务规则

落在推理路径上的哪里：报告不只是三个系数，还要按 `inspection` 段做判定：`slope_vs_nominal` 看 a1 相对标称值的偏离百分比有没有超过 `slope_tolerance_pct`，`residual_chi2` 看卡方有没有超过 `max_residual_chi2`，两项都过才是 ACCEPT，有一项不过就是 REJECT，`checks` 里逐项给出 value / limit / passed。判定与拟合口径错一个，放行结论就反了。原实现完全没有这一段。

对应测试：test_verdict_accept_when_within_limits、test_verdict_rejects_off_nominal_slope、test_verdict_rejects_bad_residual_chi2

## 难点 5：边界值与拒收的副作用

推理轴：边界条件

落在推理路径上的哪里：有效点恰好等于 `min_used` 必须照算，少一个就必须拒收；`plateau_tolerance_c` 取 0、sigma 非正、点数不足十二、没有温度跨度都算非法记录。拒收时退出码是 2、stderr 要有说明、不落证书、不留临时文件。原实现没有任何校验，什么输入都写一份报告。

对应测试：test_exactly_min_used_points_accepted、test_one_minus_min_used_rejected、test_zero_plateau_tolerance_rejected、test_invalid_data_rejected_without_output

## 跨运行状态（重复运行与不污染）

同一份台账重复运行，证书逐字节一致（不能写生成时间）；失败的那次不能改写上一张证书：先写临时文件再原子替换，非法记录以退出码 2 收场且不落文件。对应测试：test_rerun_produces_identical_bytes、test_failed_run_keeps_previous_report。输出契约另有 test_report_schema_and_rounding 覆盖字段、位数与有限性。

## 权重口径

19 条计分测试 + 1 条门禁（test_correlated_errors_use_gls，不通过则整份记 0）：核心系数 0.085、标准误 0.075、恒温段剔除 0.045、第二套数据 0.055、卡方 0.04、相关系数 0.04、GLS 标准误 0.09、共模误差 0.09、min_used 等值 0.08、少一个点 0.08、大不确定度点 0.07、非法记录 0.06、输出契约 0.02、重复运行 0.02、失败不覆盖 0.01、恒温容差为 0 的边界 0.025、判定 ACCEPT 0.04、斜率超差 0.04、残差超限 0.035，和为 1。

## 定标依据（自检用）

判分侧独立按加权 / 广义最小二乘复算（自己的 3×3 求解、求逆与 Cholesky 白化），与参考解在系数上一致到 1e-9、标准误与相关系数一致到 1e-12。噪声按 std 恰好等于 sigma 生成，所以出厂记录的 `residual_chi2` 落在 1 附近（实测 1.034）。本地实测参考解 1.000000、缺陷版 0.000000；缺陷版是等权最小二乘、把恒温段全算进去、没有不确定度没有判定、报告带时间戳，20 条全挂。
