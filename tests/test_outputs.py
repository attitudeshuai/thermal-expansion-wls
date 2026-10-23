#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判分输出：10 条行为测试，按重要程度加权，权重和 = 1。

每条测试自己造一段升温记录，按加权最小二乘独立复算系数、标准误、相关系数与卡方，
再驱动被测程序的命令行入口比对报告。期望值不读被测代码，也不读参考解的输出。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

APP = os.environ.get("CT_APP_DIR", "/app")
ENTRY = os.path.join(APP, "src", "expansion.py")

WEIGHTS = {
    "test_coefficients_match_independent_wls": 0.09,
    "test_std_errors_reflect_point_weights": 0.07,
    "test_plateau_run_kept_once_only": 0.06,
    "test_second_dataset_with_other_range": 0.05,
    "test_chi2_reduced_near_unity": 0.04,
    "test_corr_a1_a2_matches_independent": 0.04,
    "test_correlated_errors_use_gls": 0.09,
    "test_correlated_errors_std_error_matches_gls": 0.10,
    "test_common_mode_error_handled": 0.10,
    "test_exactly_min_used_points_accepted": 0.09,
    "test_one_minus_min_used_rejected": 0.09,
    "test_huge_sigma_point_barely_moves_fit": 0.08,
    "test_invalid_data_rejected_without_output": 0.05,
    "test_report_schema_and_rounding": 0.02,
    "test_rerun_produces_identical_bytes": 0.02,
    "test_failed_run_keeps_previous_report": 0.01,
}


class _Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_data(a1=0.600, a2=0.0008, a0=0.0, t_ref=25.0, t_start=25.0, t_end=225.0,
              step=1.0, plateau_at=120.0, plateau_len=40, sigma_lo=0.003, sigma_hi=0.003,
              rho=0.0, common_mode=0.0, skew=1.0, seed=101):
    rng = _Lcg(seed)
    temps = []
    t = t_start
    frac = 0.0
    while t <= t_end + 1e-9:
        temps.append(round(t, 2))
        frac = min(frac + 1.0 / 400.0, 1.0)
        t += step * (1.0 + (skew - 1.0) * frac)
    plateau = [round(plateau_at, 2)] * plateau_len
    pos = min(range(len(temps)), key=lambda k: abs(temps[k] - plateau_at))
    temps = temps[:pos] + plateau + temps[pos:]
    span = max(temps) - min(temps)
    # 相关噪声：e_i = sqrt(1-ρ²)·ρ^i·g_0 + 白噪声（相邻误差按 ρ 衰减）
    gauss = [(rng.next_u() + rng.next_u() - 1.0) * math.sqrt(6.0) for _ in temps]
    shared = ((rng.next_u() + rng.next_u() - 1.0) * math.sqrt(6.0)) * common_mode
    samples = []
    common = 0.0
    for idx, temp in enumerate(temps):
        frac = (temp - min(temps)) / span if span else 0.0
        sigma = sigma_lo + (sigma_hi - sigma_lo) * frac
        dT = temp - t_ref
        if rho > 0:
            common = rho * common + math.sqrt(max(1.0 - rho * rho, 0.0)) * gauss[idx]
            noise = common * sigma + shared
        else:
            noise = gauss[idx] * sigma
        y = a0 + a1 * dT + a2 * dT * dT + noise
        samples.append({"t": temp, "L": round(y, 6), "sigma": round(sigma, 6)})
    return {"unit": {"temperature": "degC", "length": "um"},
            "reference_temperature": t_ref, "samples": samples,
            "fit": {"plateau_tolerance_c": 0.02, "min_used": 10,
                    "error_correlation": rho, "common_mode_sigma": common_mode}}


def _keep(rows, tol):
    kept = []
    for r in rows:
        if kept and abs(r["t"] - kept[-1]["t"]) < tol:
            continue
        kept.append(r)
    return kept


def _solve3(a, b):
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for k in range(col, 4):
            m[col][k] /= pv
        for r in range(3):
            if r != col:
                f = m[r][col]
                for k in range(col, 4):
                    m[r][k] -= f * m[col][k]
    return [m[0][3], m[1][3], m[2][3]]


def _inv3(a):
    inv = [[0.0] * 3 for _ in range(3)]
    for j in range(3):
        col = _solve3(a, [1.0 if i == j else 0.0 for i in range(3)])
        for i in range(3):
            inv[i][j] = col[i]
    return inv



def _chol(mat):
    n = len(mat)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = mat[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                if acc <= 0:
                    return None
                low[i][i] = math.sqrt(acc)
            else:
                low[i][j] = acc / low[j][j]
    return low


def _fsub(low, vec):
    n = len(vec)
    out = [0.0] * n
    for i in range(n):
        acc = vec[i] - sum(low[i][k] * out[k] for k in range(i))
        out[i] = acc / low[i][i]
    return out


def _whiten(cols, sigmas, rho, common=0.0):
    n = len(sigmas)
    if rho <= 0 and common <= 0:
        return [[c[i] / sigmas[i] for i in range(n)] for c in cols]
    cov = [[sigmas[i] * sigmas[j] * (rho ** abs(i - j)) + common * common
            for j in range(n)] for i in range(n)]
    low = _chol(cov)
    if low is None:
        return None
    return [_fsub(low, list(c)) for c in cols]

def expect(data):
    t_ref = float(data["reference_temperature"])
    rows = _keep([{"t": float(s["t"]), "y": float(s["L"]), "s": float(s["sigma"])}
                  for s in data["samples"]], float(data["fit"]["plateau_tolerance_c"]))
    rho = float((data.get("fit") or {}).get("error_correlation", 0.0) or 0.0)
    common = float((data.get("fit") or {}).get("common_mode_sigma", 0.0) or 0.0)
    sigmas = [r["s"] for r in rows]
    cols = [[1.0 for _ in rows],
            [r["t"] - t_ref for r in rows],
            [(r["t"] - t_ref) ** 2 for r in rows],
            [r["y"] for r in rows]]
    white = _whiten(cols, sigmas, rho, common)
    if white is None:
        raise AssertionError("相关系数导致协方差矩阵不正定")
    w0, w1, w2, wy = white
    xtwx = [[0.0] * 3 for _ in range(3)]
    xtwy = [0.0] * 3
    for i in range(len(rows)):
        x = [w0[i], w1[i], w2[i]]
        for a in range(3):
            xtwy[a] += x[a] * wy[i]
            for b in range(3):
                xtwx[a][b] += x[a] * x[b]
    beta = _solve3(xtwx, xtwy)
    sse = 0.0
    for i in range(len(rows)):
        m = beta[0] * w0[i] + beta[1] * w1[i] + beta[2] * w2[i]
        sse += (wy[i] - m) ** 2
    dof = len(rows) - 3
    var = sse / dof
    cov = [[var * v for v in row] for row in _inv3(xtwx)]
    se = [math.sqrt(max(cov[i][i], 0.0)) for i in range(3)]
    corr = cov[1][2] / math.sqrt(cov[1][1] * cov[2][2])
    return {"beta": beta, "se": se, "corr": corr, "chi2": sse / dof,
            "fitted_points": len(rows), "degrees_of_freedom": dof, "rho": rho}


def _run(path, out_path, timeout=240):
    cmd = [sys.executable, ENTRY, "--data", path, "--out", out_path]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def run_case(data):
    tmp = tempfile.mkdtemp(prefix="ex-case-")
    dpath = os.path.join(tmp, "dilatometer.json")
    opath = os.path.join(tmp, "report.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    proc = _run(dpath, opath)
    report = None
    if os.path.exists(opath):
        try:
            with open(opath, encoding="utf-8") as f:
                report = json.load(f)
        except ValueError:
            report = None
    return report, {"data": dpath, "out": opath, "proc": proc, "dir": tmp}


def _close(got, want, rel, floor=0.0):
    try:
        v = float(got)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return abs(v - want) <= max(rel * abs(want), floor)


def _complete(report):
    need = {"recorded_points", "fitted_points", "degrees_of_freedom", "reference_temperature", "expansion_terms",
            "term_uncertainties", "slope_intercept_correlation", "residual_chi2"}
    if not isinstance(report, dict) or not need.issubset(set(report)):
        return False
    for group in ("expansion_terms", "term_uncertainties"):
        for key in ("intercept_um", "slope_um_per_c", "curvature_um_per_c2"):
            try:
                if not math.isfinite(float(report[group][key])):
                    return False
            except (KeyError, TypeError, ValueError):
                return False
    return True


def test_coefficients_match_independent_wls():
    """题面原文：「按加权最小二乘拟合 L = a0 + a1·ΔT + a2·ΔT²」。

    出厂记录：三个系数与独立复算相差不超过 0.5%。
    """
    data = make_data(seed=101)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    if not _complete(report):
        return False
    exp = expect(data)
    coef = report.get("expansion_terms") or {}
    return (_close(coef.get("intercept_um"), exp["beta"][0], 0.0015, 5e-6)
            and _close(coef.get("slope_um_per_c"), exp["beta"][1], 0.0015)
            and _close(coef.get("curvature_um_per_c2"), exp["beta"][2], 0.0015))


def test_std_errors_reflect_point_weights():
    """题面原文：「权重取 1/sigma²，参数标准误由加权协方差给出」。

    逐点不确定度从 0.002 涨到 0.05 的记录：a1 的标准误必须与加权复算一致（5% 以内）。
    把 sigma 当等权处理时，标准误会差出好几倍。
    """
    data = make_data(sigma_lo=0.002, sigma_hi=0.05, seed=211)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = expect(data)
    se = report.get("term_uncertainties") or {}
    return _close(se.get("slope_um_per_c"), exp["se"][1], 0.05)


def test_plateau_run_kept_once_only():
    """题面原文：「连续恒温保持点只保留第一个，其余剔除」。

    记录里有 40 个 120 °C 的保持点：n_used 必须把重复点剔掉，系数不受影响。
    """
    data = make_data(plateau_at=120.0, plateau_len=40, seed=307)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = expect(data)
    return (report.get("fitted_points") == exp["fitted_points"]
            and report.get("recorded_points") == len(data["samples"])
            and _close((report.get("expansion_terms") or {}).get("slope_um_per_c"), exp["beta"][1], 0.0015))


def test_second_dataset_with_other_range():
    """题面原文：「复检会喂别的升温范围与膨胀系数」。

    换成 40→400 °C、a1=1.6、a2=0.004 的记录后仍要吻合。
    """
    data = make_data(a1=1.6, a2=0.004, t_start=40.0, t_end=400.0, step=2.0,
                     plateau_at=200.0, plateau_len=20, sigma_lo=0.004, sigma_hi=0.02,
                     seed=409)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    if not _complete(report):
        return False
    exp = expect(data)
    coef = report.get("expansion_terms") or {}
    return (_close(coef.get("slope_um_per_c"), exp["beta"][1], 0.0015)
            and _close(coef.get("curvature_um_per_c2"), exp["beta"][2], 0.0015))


def test_chi2_reduced_near_unity():
    """题面原文：「chi2_red = 加权残差平方和 /(n-3)」。

    出厂记录的噪声按 sigma 生成，chi2_red 应在 1 附近（0.4 到 2.0），且与独立复算相差不超过 5%。
    """
    data = make_data(seed=503)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = expect(data)
    try:
        got = float(report["residual_chi2"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0.4 <= got <= 2.0 and _close(got, exp["chi2"], 0.05)


def test_corr_a1_a2_matches_independent():
    """题面原文：「给出 a1 与 a2 的相关系数」。

    二次项与一次项高度负相关：相关系数与独立复算相差不超过 0.01（绝对值）。
    """
    data = make_data(seed=601)
    report, _ = run_case(data)
    if not isinstance(report, dict):
        return False
    exp = expect(data)
    try:
        got = float(report["slope_intercept_correlation"])
    except (KeyError, TypeError, ValueError):
        return False
    return abs(got - exp["corr"]) <= 0.01


def test_correlated_errors_use_gls():
    """题面原文：「fit.error_correlation 给出相邻误差的相关系数 ρ……非零时是广义最小二乘」。

    ρ=0.6 的数据上，报告斜率必须与独立 GLS 相差不超过 1%，并且报告里回填了 error_correlation。
    把相关误差当成独立误差（普通加权最小二乘）会明显偏。
    """
    data = make_data(sigma_lo=0.004, sigma_hi=0.010, rho=0.6, seed=211)
    report, _ = run_case(data)
    exp = expect(data)
    a1 = float(report["expansion_terms"]["slope_um_per_c"])
    assert abs(a1 - exp["beta"][1]) <= 0.01 * abs(exp["beta"][1]), \
        "GLS 斜率没对上：被测 %r，独立复算 %r" % (a1, exp["beta"][1])
    chi2 = float(report["residual_chi2"])
    assert abs(chi2 - exp["chi2"]) <= 0.03 * exp["chi2"], \
        "chi2_red 没在白化后的残差上算：被测 %r，独立复算 %r" % (chi2, exp["chi2"])
    assert abs(float(report["error_correlation"]) - 0.6) <= 1e-9, \
        "报告没回填 error_correlation：%r" % report.get("error_correlation")
    return True


def test_correlated_errors_std_error_matches_gls():
    """题面原文：「参数协方差 = sigma_hat²·(XᵀWX)⁻¹」——相关误差下 W 换成 Σ⁻¹。

    ρ=0.6 时斜率的标准不确定度与独立 GLS 相差不超过 2%，且 chi2_red 仍在噪声量级。
    """
    data = make_data(sigma_lo=0.004, sigma_hi=0.010, rho=0.6, seed=307)
    report, _ = run_case(data)
    exp = expect(data)
    se1 = float(report["term_uncertainties"]["slope_um_per_c"])
    assert abs(se1 - exp["se"][1]) <= 0.02 * abs(exp["se"][1]), \
        "GLS 斜率标准误没对上：被测 %r，独立复算 %r" % (se1, exp["se"][1])
    wls = expect(make_data(sigma_lo=0.004, sigma_hi=0.010, rho=0.0, seed=307))
    assert se1 > 1.3 * wls["se"][1], \
        "相关误差下标准误没放大（当成独立误差了）：被测 %r，独立误差基线 %r" % (se1, wls["se"][1])
    assert float(report["residual_chi2"]) <= 2.0, "chi2_red 过大：%r" % report["residual_chi2"]
    return True


def test_common_mode_error_handled():
    """题面原文：「协方差矩阵是 Σ = D·R·D + τ²·J」。

    叠上 τ=0.0025 的共模误差：斜率要落在独立复算的 0.3% 以内，且报告回填了 common_mode_sigma。
    只按相关项白化、漏掉共模项的实现，标准误会被低估。
    """
    data = make_data(sigma_lo=0.002, sigma_hi=0.004, common_mode=0.0025, seed=421)
    report, _ = run_case(data)
    exp = expect(data)
    slope = float(report["expansion_terms"]["slope_um_per_c"])
    se = float(report["term_uncertainties"]["slope_um_per_c"])
    assert abs(slope - exp["beta"][1]) <= 0.003 * abs(exp["beta"][1]), \
        "共模工况下滑率没对上：被测 %r，独立复算 %r" % (slope, exp["beta"][1])
    assert abs(se - exp["se"][1]) <= 0.05 * abs(exp["se"][1]), \
        "共模工况下标准误没对上：被测 %r，独立复算 %r" % (se, exp["se"][1])
    assert float(report["common_mode_sigma"]) == 0.0025, "报告没回填 common_mode_sigma"
    return True


def test_exactly_min_used_points_accepted():
    """题面原文：「剔除后有效点少于 min_used 才拒收，等于 min_used 必须照算」。

    构造恰好剩 min_used 个有效点的台账：程序要出报告，不能因为「等于」被判成「少于」。
    """
    data = make_data(t_start=25.0, t_end=45.0, step=1.0, plateau_at=35.0, plateau_len=6, seed=701)
    first, _ = run_case(data)
    assert isinstance(first, dict), "基线台账没出报告"
    usable = int(first["fitted_points"])
    data["fit"]["min_used"] = usable
    report, _ = run_case(data)
    assert isinstance(report, dict), "恰好 min_used=%d 个点时却拒收了" % usable
    assert int(report["fitted_points"]) == usable, "拟合点数变了：%r" % report.get("fitted_points")
    return True


def test_one_minus_min_used_rejected():
    """题面原文：「剔除后有效点少于 min_used 必须拒收」。

    同一份台账把 min_used 抬到比可用点还多一个：必须退出码 2 且不产出证书。
    """
    data = make_data(t_start=25.0, t_end=45.0, step=1.0, plateau_at=35.0, plateau_len=6, seed=703)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "基线台账没出报告"
    usable = int(report["fitted_points"])
    data["fit"]["min_used"] = usable + 1
    report2, ctx2 = run_case(data)
    assert report2 is None, "可用点不足 min_used 时仍产出了证书"
    assert ctx2["proc"].returncode == 2, "退出码应为 2，实际 %r" % ctx2["proc"].returncode
    return True


def test_huge_sigma_point_barely_moves_fit():
    """题面原文：「每个温度点的重复测量标准差也一并记录，越往高温噪声越大」。

    把一个点的不确定度放大 60 倍：加权后它几乎不起作用，斜率应与去掉该点相差不超过 0.05%。
    """
    data = make_data(sigma_lo=0.002, sigma_hi=0.004, seed=709)
    keep = make_data(sigma_lo=0.002, sigma_hi=0.004, seed=709)
    keep["samples"] = [s for s in keep["samples"] if s["t"] != 120.0]
    boom = make_data(sigma_lo=0.002, sigma_hi=0.004, seed=709)
    for s in boom["samples"]:
        if s["t"] == 120.0:
            s["sigma"] = round(s["sigma"] * 60.0, 6)
    rep_keep, _ = run_case(keep)
    rep_boom, _ = run_case(boom)
    a = float(rep_keep["expansion_terms"]["slope_um_per_c"])
    b = float(rep_boom["expansion_terms"]["slope_um_per_c"])
    assert abs(a - b) <= 0.0005 * abs(a), "大不确定度的点权重没压下去：%r 对 %r" % (b, a)
    return True


def test_invalid_data_rejected_without_output():
    """题面原文：「记录不合法时以退出码 2 结束，不产出报告」。

    四种坏记录：sigma 非正、采样点不足 12、没有温度跨度、剔除后有效点太少。
    """
    good = make_data(seed=701)
    bad_sigma = json.loads(json.dumps(good))
    bad_sigma["samples"][5]["sigma"] = 0.0
    few = json.loads(json.dumps(good))
    few["samples"] = few["samples"][:8]
    flat = json.loads(json.dumps(good))
    for s in flat["samples"]:
        s["t"] = 25.0
    short = json.loads(json.dumps(good))
    short["fit"]["min_used"] = 250
    for data in (bad_sigma, few, flat, short):
        tmp = tempfile.mkdtemp(prefix="ex-bad-")
        dpath = os.path.join(tmp, "dilatometer.json")
        opath = os.path.join(tmp, "report.json")
        with open(dpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        proc = _run(dpath, opath)
        if proc.returncode != 2 or os.path.exists(opath) or not (proc.stderr or "").strip():
            return False
        if [n for n in os.listdir(tmp) if n.endswith(".tmp")]:
            return False
    return True


def test_report_schema_and_rounding():
    """题面原文：「报告结构固定，系数与标准误保留六位小数，相关系数与卡方三位」。

    校验顶层字段、位数与有限性，文件里不出现 NaN 或 Infinity。
    """
    data = make_data(seed=809)
    report, ctx = run_case(data)
    if not isinstance(report, dict):
        return False
    need = {"recorded_points", "fitted_points", "degrees_of_freedom", "reference_temperature", "expansion_terms",
            "term_uncertainties", "slope_intercept_correlation", "residual_chi2", "certificate_note"}
    if not need.issubset(set(report)):
        return False
    for group in ("expansion_terms", "term_uncertainties"):
        for key in ("intercept_um", "slope_um_per_c", "curvature_um_per_c2"):
            v = float(report[group][key])
            if not math.isfinite(v) or abs(v - round(v, 6)) > 1e-12:
                return False
    for key in ("slope_intercept_correlation", "residual_chi2"):
        v = float(report[key])
        if not math.isfinite(v) or abs(v - round(v, 3)) > 1e-12:
            return False
    if report.get("degrees_of_freedom") != report.get("fitted_points") - 3:
        return False
    raw = open(ctx["out"], encoding="utf-8").read()
    return "NaN" not in raw and "Infinity" not in raw


def test_rerun_produces_identical_bytes():
    """题面原文：「同一份记录重复运行，报告逐字节一致」。

    连跑两次，两个报告文件字节完全相同（报告里不能带运行时间戳）。
    """
    data = make_data(seed=907)
    first, ctx1 = run_case(data)
    if first is None:
        return False
    if not {"fitted_points", "degrees_of_freedom", "residual_chi2"}.issubset(set(first)):
        return False
    body1 = open(ctx1["out"], "rb").read()
    second, ctx2 = run_case(data)
    if second is None:
        return False
    return body1 == open(ctx2["out"], "rb").read()


def test_failed_run_keeps_previous_report():
    """题面原文：「运行失败时不能覆盖上一次已经写好的报告」。

    先跑一次合格记录留下报告，再用 sigma 非正的坏记录跑：退出码 2，旧报告字节不变。
    """
    good = make_data(seed=1009)
    tmp = tempfile.mkdtemp(prefix="ex-keep-")
    dpath = os.path.join(tmp, "dilatometer.json")
    opath = os.path.join(tmp, "report.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False)
    first = _run(dpath, opath)
    if first.returncode != 0 or not os.path.exists(opath):
        return False
    before = open(opath, "rb").read()
    broken = json.loads(json.dumps(good))
    broken["samples"][3]["sigma"] = -1.0
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    second = _run(dpath, opath)
    if second.returncode != 2:
        return False
    return open(opath, "rb").read() == before


def main():
    total = 0.0
    for name, weight in WEIGHTS.items():
        fn = globals().get(name)
        try:
            passed = bool(fn and fn())
        except Exception as e:  # noqa: BLE001 —— 任何异常都按不通过计
            print("%s: 异常 %r" % (name, e), file=sys.stderr)
            passed = False
        total += weight * passed
        print("%s: %s (权重 %.2f)" % (name, "PASS" if passed else "FAIL", weight), file=sys.stderr)
    print("%.6f" % total)


if __name__ == "__main__":
    main()
