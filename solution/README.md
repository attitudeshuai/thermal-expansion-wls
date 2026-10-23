# 难点与测试对应

## 难点 1：异方差必须加权，权重是 1/sigma²

推理轴：领域公式（加权最小二乘）

落在推理路径上的哪里：每个采样点自带不确定度 sigma，升温越高噪声越大。权重取 1/sigma² 才是最大似然；等权处理会把高温端的高噪声点和低温端的低噪声点同等对待，系数与标准误一起偏。原实现连 sigma 都没读。

对应测试：test_coefficients_match_independent_wls、test_std_errors_reflect_point_weights、test_chi2_reduced_near_unity、test_huge_sigma_point_barely_moves_fit

## 难点 2：相变台阶是反问题，不能当二次曲线抹平

推理轴：领域知识 + 非线性反演

落在推理路径上的哪里：合金过相变时长度多出一个台阶 `A·S((T − Tc)/w)`，台阶的陡缓（`w`）与位置（`Tc`）因炉次而异，记录里不给。把台阶当曲率抹平，二次项会被拽偏，`Tc` 整个丢掉；要用「白化残差平方和最小」把三个非线性参数从数据里反演出来。给定 `Tc`、`w` 时 `[a0, a1, a2, A]` 是线性的，所以能先粗扫再细化，但粗扫网格与细化顺序必须真的收敛到同一点。

对应测试：test_transition_terms_match_independent、test_second_dataset_with_other_range

## 难点 3：不确定度要把非线性参数的雅可比算进去

推理轴：数值方法（Gauss-Newton 协方差）

落在推理路径上的哪里：参数协方差 = `sigma_hat²·(JᵀJ)⁻¹`，`J` 必须包含台阶对 `Tc` 与对 `w` 的两列偏导（`A·S(1-S)·(-1/w)` 与 `A·S(1-S)·(-(T-Tc)/w²)`）。把 `Tc`、`w` 当成已知量、只对线性参数求协方差，相变参数的标准误会明显偏小、整份不确定度预算就是错的。

对应测试：test_transition_uncertainties_reported、test_corr_a1_a2_matches_independent

## 难点 4：相关误差与共模漂移要走广义最小二乘

推理轴：数值方法（Cholesky 白化）

落在推理路径上的哪里：`fit.error_correlation` 非零或 `fit.common_mode_sigma` 非零时，各点误差不再独立，协方差矩阵是 `Σ = D·R·D + τ²·J`。这时要整体做 Cholesky 分解、把设计列与观测一起前代白化，再解同一个正规方程（非线性反演也必须在白化空间里做）。把相关误差当成独立误差，斜率会偏、标准误偏小。

对应测试：test_correlated_errors_use_gls（门禁）、test_correlated_errors_std_error_matches_gls、test_common_mode_error_handled

## 难点 5：恒温保持段只留第一个点

推理轴：数据口径

落在推理路径上的哪里：控温平台有 40 个重复温度点，它们不带来新的温度信息，却把权重全堆在同一个温度上。规则是每组连续保持点只保留第一个，其余剔除。原实现把这些点全算进去了。

对应测试：test_plateau_run_kept_once_only

## 难点 6：首件核查判定（业务规则）

推理轴：业务规则

落在推理路径上的哪里：报告不只是参数，还要按 `inspection` 段做三项判定：`slope_vs_nominal` 看 a1 相对标称值的偏离百分比有没有超过 `slope_tolerance_pct`，`residual_chi2` 看卡方有没有超过 `max_residual_chi2`，`midpoint_vs_nominal` 看 `Tc` 与标称相变中点的偏差有没有超过 `midpoint_tolerance_c`；三项都过才是 ACCEPT，有一项不过就是 REJECT，`checks` 里逐项给出 value / limit / passed。原实现完全没有这一段。

对应测试：test_verdict_accept_when_within_limits、test_verdict_rejects_off_nominal_slope、test_verdict_rejects_bad_residual_chi2、test_verdict_rejects_off_nominal_midpoint

## 难点 7：边界值与拒收的副作用

推理轴：边界条件

落在推理路径上的哪里：有效点恰好等于 `min_used` 必须照算，少一个就必须拒收；`plateau_tolerance_c` 取 0、sigma 非正、点数不足十二、没有温度跨度都算非法记录。拒收时退出码是 2、stderr 要有说明、不落证书、不留临时文件。

对应测试：test_exactly_min_used_points_accepted、test_one_minus_min_used_rejected、test_zero_plateau_tolerance_rejected、test_invalid_data_rejected_without_output

## 跨运行状态（重复运行与不污染）

同一份台账重复运行，证书逐字节一致（不能写生成时间）；失败的那次不能改写上一张证书：先写临时文件再原子替换，非法记录以退出码 2 收场且不落文件。对应测试：test_rerun_produces_identical_bytes、test_failed_run_keeps_previous_report。输出契约另有 test_report_schema_and_rounding 覆盖字段、位数与有限性。

## 权重口径

22 条计分测试 + 1 条门禁（`test_correlated_errors_use_gls`，不通过则整份记 0），权重和 = 1。权重最高的三项是 相变台阶反演 0.074、相变不确定度 0.071、GLS 标准误 0.071；逐条权重与理由写在 `tests/scoring.json` 里，其余各项按领域重要性递减分配：核心反演与不确定度 > 判定与边界 > 输出契约与重复运行。

## 定标依据（自检用）

判分侧另写一套反演：自己的 (Tc, w) 网格（21×8，与参考解的 25×10 不同）、自己的黄金分割细化顺序与 6 列雅可比，同一份记录上两者在系数上一致到 0.2%、相变中点一致到 0.05%、相变参数标准误一致到 5% 以内。噪声按 std 恰好等于 sigma 生成，所以出厂记录的 `residual_chi2` 落在 1 附近。本地实测参考解 1.000000、缺陷版 0.000000；缺陷版是普通最小二乘、没有台阶、没有不确定度、没有判定、报告带时间戳，26 条全挂。
