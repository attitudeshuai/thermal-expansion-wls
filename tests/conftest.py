# -*- coding: utf-8 -*-
"""pytest 汇总钩子：按权重把每条测试的结果折算成加权分，写进 weighted.txt。

口径：每条测试占一个权重，权重和 = 1（见 WEIGHTS）；只有 PASS 才拿到该条权重；
不在 WEIGHTS 里的测试名一律不给分（防止误加测试把总分抬高）。
这里只写中间结果 weighted.txt（本地调试可用 CT_WEIGHTED_PATH 覆盖），
reward.txt 由 tests/test.sh 在确认 pytest 结构化报告非空之后再落。
"""
import os

WEIGHTS = {
    "test_std_errors_reflect_point_weights": 0.088607594,
    "test_second_dataset_with_other_range": 0.063291139,
    "test_chi2_reduced_near_unity": 0.050632911,
    "test_corr_a1_a2_matches_independent": 0.050632911,
    "test_correlated_errors_std_error_matches_gls": 0.126582279,
    "test_common_mode_error_handled": 0.126582279,
    "test_exactly_min_used_points_accepted": 0.113924051,
    "test_one_minus_min_used_rejected": 0.113924051,
    "test_huge_sigma_point_barely_moves_fit": 0.101265822,
    "test_report_schema_and_rounding": 0.025316456,
    "test_rerun_produces_identical_bytes": 0.025316456,
    "test_failed_run_keeps_previous_report": 0.012658228,
    "test_coefficients_match_independent_wls": 0.101265823,
}

_RESULTS = {}


def pytest_runtest_logreport(report):
    if report.when in ("setup", "call") and report.failed:
        _RESULTS[report.nodeid.split("::")[-1]] = False
    elif report.when == "call" and report.passed:
        _RESULTS.setdefault(report.nodeid.split("::")[-1], True)


def pytest_sessionfinish(session, exitstatus):
    # 完成报告校验：必须有每一条测试的结果，缺任何一条都按 INCOMPLETE 记 0
    missing = [n for n in WEIGHTS if n not in _RESULTS]
    total = 0.0
    if missing:
        print("[INCOMPLETE] 这些测试没有跑完：%s；记 0" % ", ".join(missing))
    else:
        for name, weight in WEIGHTS.items():
            if _RESULTS.get(name) is True:
                total += weight
    path = os.environ.get("CT_WEIGHTED_PATH", "/logs/verifier/weighted.txt")
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("%.6f\n" % total)
