#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判分输出：二十二条行为测试按重要程度加权，权重和 = 1，另有一条门禁用例。

每条测试自己造一段升温记录，按加权 / 广义最小二乘独立复算系数、相变台阶参数、
标准误、相关系数与卡方，再驱动被测程序的命令行入口比对证书。期望值不读被测代码，
也不读参考解的输出。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import pytest

APP = os.environ.get("CT_APP_DIR", "/app")
ENTRY = os.path.join(APP, "src", "expansion.py")
NPAR = 6


class _Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def _logistic(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def make_data(a1=0.600, a2=0.0008, a0=0.0, a_step=2.2, tc=180.0, width=1.0,
              t_ref=25.0, t_start=25.0, t_end=400.0, step=1.0, plateau_at=120.0,
              plateau_len=40, sigma_lo=0.003, sigma_hi=0.006, rho=0.0, common_mode=0.0,
              seed=101):
    rng = _Lcg(seed)
    temps = []
    t = t_start
    while t <= t_end + 1e-9:
        temps.append(round(t, 2))
        t += step
    plateau = [round(plateau_at, 2)] * plateau_len
    pos = min(range(len(temps)), key=lambda k: abs(temps[k] - plateau_at))
    temps = temps[:pos] + plateau + temps[pos:]
    span = max(temps) - min(temps)
    # 相关噪声：common 是相邻误差按 ρ 衰减的那部分，shared 是全部点共模漂移
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
        y = a0 + a1 * dT + a2 * dT * dT + a_step * _logistic((temp - tc) / width) + noise
        samples.append({"t": temp, "L": round(y, 6), "sigma": round(sigma, 6)})
    return {"unit": {"temperature": "degC", "length": "um"},
            "reference_temperature": t_ref, "samples": samples,
            "inspection": {"slope_nominal_um_per_c": a1, "slope_tolerance_pct": 1.0,
                           "max_residual_chi2": 1.5,
                           "midpoint_nominal_c": tc, "midpoint_tolerance_c": 3.0},
            "fit": {"plateau_tolerance_c": 0.02, "min_used": 12,
                    "error_correlation": rho, "common_mode_sigma": common_mode}}


def _keep(rows, tol):
    kept = []
    for r in rows:
        if kept and abs(r["t"] - kept[-1]["t"]) < tol:
            continue
        kept.append(r)
    return kept


def _solve(a, b):
    n = len(a)
    m = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-14:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for k in range(col, n + 1):
            m[col][k] /= pv
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f:
                for k in range(col, n + 1):
                    m[r][k] -= f * m[col][k]
    return [m[i][n] for i in range(n)]


def _inv(a):
    n = len(a)
    out = [[0.0] * n for _ in range(n)]
    for j in range(n):
        col = _solve(a, [1.0 if i == j else 0.0 for i in range(n)])
        if col is None:
            return None
        for i in range(n):
            out[i][j] = col[i]
    return out


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


def _whitener(sigmas, rho, common):
    n = len(sigmas)
    if rho <= 0 and common <= 0:
        return lambda vec: [vec[i] / sigmas[i] for i in range(n)]
    cov = [[sigmas[i] * sigmas[j] * (rho ** abs(i - j)) + common * common
            for j in range(n)] for i in range(n)]
    low = _chol(cov)
    if low is None:
        return None
    return lambda vec: _fsub(low, list(vec))


def _normals(cols, y):
    npar = len(cols)
    xtx = [[0.0] * npar for _ in range(npar)]
    xty = [0.0] * npar
    for i in range(len(y)):
        for a in range(npar):
            ca = cols[a][i]
            xty[a] += ca * y[i]
            for b in range(npar):
                xtx[a][b] += ca * cols[b][i]
    return xtx, xty


def _golden(fn, lo, hi, iters=70):
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def expect(data):
    """独立复算：自己的一套 (Tc, w) 网格与细化路径（网格与细化顺序和参考解不同）。"""
    t_ref = float(data["reference_temperature"])
    fit_cfg = data["fit"] or {}
    rows = _keep([{"t": float(s["t"]), "y": float(s["L"]), "s": float(s["sigma"])}
                  for s in data["samples"]], float(fit_cfg["plateau_tolerance_c"]))
    rho = float(fit_cfg.get("error_correlation", 0.0) or 0.0)
    common = float(fit_cfg.get("common_mode_sigma", 0.0) or 0.0)
    sigmas = [r["s"] for r in rows]
    dt = [r["t"] - t_ref for r in rows]
    ts = [r["t"] for r in rows]
    ys = [r["y"] for r in rows]
    whiten = _whitener(sigmas, rho, common)
    if whiten is None:
        raise AssertionError("相关系数导致协方差矩阵不正定")
    wx0 = whiten([1.0] * len(rows))
    wx1 = whiten(dt)
    wx2 = whiten([v * v for v in dt])
    wy = whiten(ys)
    base = [wx0, wx1, wx2]

    def sse_at(tc, width):
        step = whiten([_logistic((v - tc) / width) for v in ts])
        cols = base + [step]
        xtx, xty = _normals(cols, wy)
        coef = _solve(xtx, xty)
        if coef is None:
            return None
        sse = 0.0
        for i in range(len(wy)):
            model = coef[0] * wx0[i] + coef[1] * wx1[i] + coef[2] * wx2[i] + coef[3] * step[i]
            sse += (wy[i] - model) ** 2
        return coef, sse

    lo_t = min(r["t"] for r in rows)
    hi_t = max(r["t"] for r in rows)
    span = hi_t - lo_t
    best = None
    tc_grid = [lo_t + span * (i + 1) / 22.0 for i in range(21)]
    width_grid = [0.8, 1.2, 2.0, 3.0, 4.0, 6.0, 9.0, 13.0]
    for tc in tc_grid:
        for width in width_grid:
            got = sse_at(tc, width)
            if got is None:
                continue
            coef, sse = got
            if best is None or sse < best[0]:
                best = (sse, tc, width)
    _, tc_best, w_best = best

    for _ in range(2):
        tc_best = _golden(lambda x: sse_at(x, w_best)[1], tc_best - 8.0, tc_best + 8.0)
        w_best = _golden(lambda x: sse_at(tc_best, x)[1], max(0.2, w_best / 2.5), w_best * 2.5)

    coef, sse = sse_at(tc_best, w_best)
    n = len(rows)
    dof = n - NPAR
    var = sse / dof
    step = whiten([_logistic((v - tc_best) / w_best) for v in ts])
    d_tc = [-_logistic((v - tc_best) / w_best) * (1.0 - _logistic((v - tc_best) / w_best)) / w_best
            for v in ts]
    d_w = [-_logistic((v - tc_best) / w_best) * (1.0 - _logistic((v - tc_best) / w_best))
           * (v - tc_best) / (w_best * w_best) for v in ts]
    cols = base + [step, whiten([coef[3] * v for v in d_tc]), whiten([coef[3] * v for v in d_w])]
    xtx, _ = _normals(cols, wy)
    inv = _inv(xtx)
    if inv is None:
        raise AssertionError("独立复算的雅可比奇异")
    cov = [[var * inv[i][j] for j in range(NPAR)] for i in range(NPAR)]
    se = [math.sqrt(max(cov[i][i], 0.0)) for i in range(NPAR)]
    corr = cov[1][2] / math.sqrt(cov[1][1] * cov[2][2])
    return {"beta": coef[:4], "tc": tc_best, "width": w_best, "se": se, "corr": corr,
            "chi2": sse / dof, "fitted_points": n, "degrees_of_freedom": dof, "rho": rho}


def _run(path, out_path, timeout=240):
    cmd = [sys.executable, ENTRY, "--data", path, "--out", out_path]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def run_case(data):
    tmp = tempfile.mkdtemp(prefix="te-case-")
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


def _checks(report):
    return {item["name"]: item for item in report["checks"]}


def _corrupt(data, case):
    out = json.loads(json.dumps(data))
    if case == "sigma_zero":
        out["samples"][5]["sigma"] = 0.0
    elif case == "few_points":
        out["samples"] = out["samples"][:8]
    elif case == "flat":
        for s in out["samples"]:
            s["t"] = 25.0
    else:
        out["fit"]["min_used"] = 900
    return out


def _drop_point(data, temp):
    out = json.loads(json.dumps(data))
    out["samples"] = [s for s in out["samples"] if s["t"] != temp]
    return out


def _boost_sigma(data, temp, factor):
    out = json.loads(json.dumps(data))
    for s in out["samples"]:
        if s["t"] == temp:
            s["sigma"] = round(s["sigma"] * factor, 6)
    return out


def test_coefficients_match_independent_wls():
    """题面原文：「L = a0 + a1·ΔT + a2·ΔT²」。

    出厂记录：二次项的三个系数都要与独立复算的加权拟合一致，相对差不超过 0.5%。
    """
    data = make_data(seed=101)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    terms = report["expansion_terms"]
    inter = float(terms["intercept_um"])
    slope = float(terms["slope_um_per_c"])
    curv = float(terms["curvature_um_per_c2"])
    assert abs(inter - exp["beta"][0]) <= max(0.0015 * abs(exp["beta"][0]), 5e-6), \
        "截距没对上：被测 %r，独立复算 %r" % (inter, exp["beta"][0])
    assert abs(slope - exp["beta"][1]) <= 0.0015 * abs(exp["beta"][1]), \
        "膨胀系数没对上：被测 %r，独立复算 %r" % (slope, exp["beta"][1])
    assert abs(curv - exp["beta"][2]) <= 0.0015 * abs(exp["beta"][2]), \
        "二次项没对上：被测 %r，独立复算 %r" % (curv, exp["beta"][2])


def test_transition_terms_match_independent():
    """题面原文：「A·S((T − Tc)/w)」。

    相变台阶的三项：中点、宽度、台阶高度都要与独立复算一致（中点与宽度 1%，高度 2%）。
    """
    data = make_data(seed=113)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    terms = report["transition_terms"]
    mid = float(terms["midpoint_c"])
    width = float(terms["width_c"])
    amp = float(terms["amplitude_um"])
    assert abs(mid - exp["tc"]) <= 0.01 * abs(exp["tc"]), \
        "相变中点没对上：被测 %r，独立复算 %r" % (mid, exp["tc"])
    assert abs(mid - 180.0) <= 0.01 * 180.0, "相变中点离真值太远：被测 %r，真值 180" % (mid,)
    assert abs(width - exp["width"]) <= 0.01 * abs(exp["width"]), \
        "相变宽度没对上：被测 %r，独立复算 %r" % (width, exp["width"])
    assert abs(amp - exp["beta"][3]) <= 0.02 * abs(exp["beta"][3]), \
        "台阶高度没对上：被测 %r，独立复算 %r" % (amp, exp["beta"][3])


def test_transition_uncertainties_reported():
    """题面原文：「白化空间的 6 列雅可比」。

    相变参数的三个标准误都要为正，且与独立复算相差不超过 15%；把 Tc、w 当成已知量、
    只对线性参数求协方差的写法，这三个数会明显偏小。
    """
    data = make_data(seed=127, sigma_lo=0.002, sigma_hi=0.004)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    uncs = report["transition_uncertainties"]
    se_mid = float(uncs["midpoint_c"])
    se_width = float(uncs["width_c"])
    se_amp = float(uncs["amplitude_um"])
    assert se_mid > 0.0 and se_width > 0.0 and se_amp > 0.0, \
        "相变参数的标准误不是正数：%r" % (uncs,)
    assert abs(se_mid - exp["se"][4]) <= 0.15 * exp["se"][4], \
        "中点标准误没对上：被测 %r，独立复算 %r" % (se_mid, exp["se"][4])
    assert abs(se_width - exp["se"][5]) <= 0.15 * exp["se"][5], \
        "宽度标准误没对上：被测 %r，独立复算 %r" % (se_width, exp["se"][5])
    assert abs(se_amp - exp["se"][3]) <= 0.15 * exp["se"][3], \
        "台阶高度标准误没对上：被测 %r，独立复算 %r" % (se_amp, exp["se"][3])


def test_std_errors_reflect_point_weights():
    """题面原文：「各点不确定度不同，不能当等权处理」。

    逐点不确定度从 0.002 涨到 0.05 的记录：a1 的标准误必须与加权复算一致（5% 以内）。
    """
    data = make_data(sigma_lo=0.002, sigma_hi=0.05, seed=211)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    se = float(report["term_uncertainties"]["slope_um_per_c"])
    assert abs(se - exp["se"][1]) <= 0.05 * abs(exp["se"][1]), \
        "标准误没按 1/sigma² 加权：被测 %r，独立复算 %r" % (se, exp["se"][1])


def test_plateau_run_kept_once_only():
    """题面原文：「每组连续保持点只保留第一个」。

    记录里有 40 个 120 °C 的保持点：拟合点数必须把重复点剔掉，系数不受影响。
    """
    data = make_data(plateau_at=120.0, plateau_len=40, seed=307)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    assert int(report["fitted_points"]) == exp["fitted_points"], \
        "保持点没剔干净：被测 %r，应剩 %r" % (report["fitted_points"], exp["fitted_points"])
    assert int(report["recorded_points"]) == len(data["samples"]), \
        "台账点数被改了：%r" % (report["recorded_points"],)
    slope = float(report["expansion_terms"]["slope_um_per_c"])
    assert abs(slope - exp["beta"][1]) <= 0.0015 * abs(exp["beta"][1]), \
        "剔除保持点后膨胀系数没对上：被测 %r，独立复算 %r" % (slope, exp["beta"][1])


def test_second_dataset_with_other_range():
    """题面原文：「复检会换升温范围、膨胀系数与噪声水平」。

    换成 40→260 °C、a1=1.6、相变点在 150 °C 的记录后仍要吻合。
    """
    data = make_data(a1=1.6, a2=0.004, a_step=1.4, tc=150.0, width=2.0,
                     t_start=40.0, t_end=260.0, step=1.5, plateau_at=200.0,
                     plateau_len=20, sigma_lo=0.004, sigma_hi=0.02, seed=409)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    slope = float(report["expansion_terms"]["slope_um_per_c"])
    mid = float(report["transition_terms"]["midpoint_c"])
    assert abs(slope - exp["beta"][1]) <= 0.0015 * abs(exp["beta"][1]), \
        "换量程后膨胀系数没对上：被测 %r，独立复算 %r" % (slope, exp["beta"][1])
    assert abs(mid - exp["tc"]) <= 0.01 * abs(exp["tc"]), \
        "换量程后相变中点没对上：被测 %r，独立复算 %r" % (mid, exp["tc"])
    assert int(report["degrees_of_freedom"]) == exp["degrees_of_freedom"], \
        "自由度不对：被测 %r，应为 %r" % (report["degrees_of_freedom"], exp["degrees_of_freedom"])


def test_chi2_reduced_near_unity():
    """题面原文：「chi2_red = 白化后残差平方和 /(n - 6)」。

    噪声按 sigma 生成，chi2_red 应在 1 附近（0.4 到 2.0），且与独立复算相差不超过 5%。
    """
    data = make_data(seed=503)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    got = float(report["residual_chi2"])
    assert 0.4 <= got <= 2.0, "chi2_red 不在噪声量级：%r" % (got,)
    assert abs(got - exp["chi2"]) <= 0.05 * exp["chi2"], \
        "chi2_red 没按自由度折算：被测 %r，独立复算 %r" % (got, exp["chi2"])


def test_corr_a1_a2_matches_independent():
    """题面原文：「（由协方差矩阵对应元素归一化）」。

    二次项与一次项高度负相关：相关系数与独立复算相差不超过 0.01（绝对值）。
    """
    data = make_data(seed=601)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    got = float(report["slope_intercept_correlation"])
    assert abs(got - exp["corr"]) <= 0.01, \
        "相关系数没对上：被测 %r，独立复算 %r" % (got, exp["corr"])


def test_correlated_errors_use_gls():
    """题面原文：「这时按广义最小二乘算」。

    ρ=0.6 的数据上，报告斜率必须与独立 GLS 相差不超过 1%，并且报告里回填了 error_correlation。
    """
    data = make_data(sigma_lo=0.004, sigma_hi=0.010, rho=0.6, seed=211)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    a1 = float(report["expansion_terms"]["slope_um_per_c"])
    assert abs(a1 - exp["beta"][1]) <= 0.01 * abs(exp["beta"][1]), \
        "GLS 斜率没对上：被测 %r，独立复算 %r" % (a1, exp["beta"][1])
    chi2 = float(report["residual_chi2"])
    assert abs(chi2 - exp["chi2"]) <= 0.03 * exp["chi2"], \
        "chi2_red 没在白化后的残差上算：被测 %r，独立复算 %r" % (chi2, exp["chi2"])
    assert abs(float(report["error_correlation"]) - 0.6) <= 1e-9, \
        "报告没回填 error_correlation：%r" % (report.get("error_correlation"),)


def test_correlated_errors_std_error_matches_gls():
    """题面原文：「标准误取对角线开方」。

    ρ=0.6 时斜率的标准不确定度与独立 GLS 相差不超过 2%，且 chi2_red 仍在噪声量级。
    """
    data = make_data(sigma_lo=0.004, sigma_hi=0.010, rho=0.6, seed=307)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    se1 = float(report["term_uncertainties"]["slope_um_per_c"])
    assert abs(se1 - exp["se"][1]) <= 0.02 * abs(exp["se"][1]), \
        "GLS 斜率标准误没对上：被测 %r，独立复算 %r" % (se1, exp["se"][1])
    wls = expect(make_data(sigma_lo=0.004, sigma_hi=0.010, rho=0.0, seed=307))
    assert se1 > 1.3 * wls["se"][1], \
        "相关误差下标准误没放大（当成独立误差了）：被测 %r，独立误差基线 %r" % (se1, wls["se"][1])
    assert float(report["residual_chi2"]) <= 2.0, "chi2_red 过大：%r" % (report["residual_chi2"],)


def test_common_mode_error_handled():
    """题面原文：「所有点一起漂移的共模误差标准差」。

    叠上 τ=0.0025 的共模误差：斜率要落在独立复算的 0.3% 以内，标准误在 5% 以内。
    """
    data = make_data(sigma_lo=0.002, sigma_hi=0.004, common_mode=0.0025, seed=421)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    slope = float(report["expansion_terms"]["slope_um_per_c"])
    se = float(report["term_uncertainties"]["slope_um_per_c"])
    assert abs(slope - exp["beta"][1]) <= 0.003 * abs(exp["beta"][1]), \
        "共模工况下斜率没对上：被测 %r，独立复算 %r" % (slope, exp["beta"][1])
    assert abs(se - exp["se"][1]) <= 0.05 * abs(exp["se"][1]), \
        "共模工况下标准误没对上：被测 %r，独立复算 %r" % (se, exp["se"][1])
    assert abs(float(report["common_mode_sigma"]) - 0.0025) <= 1e-9, \
        "报告没回填 common_mode_sigma：%r" % (report.get("common_mode_sigma"),)


def test_exactly_min_used_points_accepted():
    """题面原文：「合规返回 0」。

    构造恰好剩 min_used 个有效点的台账：程序要出证书，不能因为「等于」被判成「少于」。
    """
    data = make_data(t_start=25.0, t_end=70.0, step=3.5, plateau_at=40.0,
                     plateau_len=3, seed=701)
    first, ctx1 = run_case(data)
    assert isinstance(first, dict), "基线台账没出证书：stderr=%r" % (ctx1["proc"].stderr or "")
    usable = int(first["fitted_points"])
    data["fit"]["min_used"] = usable
    report, ctx2 = run_case(data)
    assert isinstance(report, dict), "恰好 min_used=%d 个点时却拒收了：stderr=%r" % (
        usable, ctx2["proc"].stderr or "")
    assert int(report["fitted_points"]) == usable, \
        "拟合点数变了：%r" % (report.get("fitted_points"),)


def test_one_minus_min_used_rejected():
    """题面原文：「剔除后有效点少于」。

    同一份台账把 min_used 抬到比可用点还多一个：必须退出码 2 且不产出证书。
    """
    data = make_data(t_start=25.0, t_end=70.0, step=3.5, plateau_at=40.0,
                     plateau_len=3, seed=703)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "基线台账没出证书：stderr=%r" % (ctx["proc"].stderr or "")
    usable = int(report["fitted_points"])
    data["fit"]["min_used"] = usable + 1
    report2, ctx2 = run_case(data)
    assert report2 is None, "可用点不足 min_used 时仍产出了证书"
    assert ctx2["proc"].returncode == 2, "退出码应为 2，实际 %r" % (ctx2["proc"].returncode,)
    assert (ctx2["proc"].stderr or "").strip(), "拒收时没在 stderr 说清缘由"
    assert not os.path.exists(ctx2["out"]), "拒收时不该留下报告文件"


def test_huge_sigma_point_barely_moves_fit():
    """题面原文：「升温越高噪声越大」。

    把一个点的不确定度放大 60 倍：加权后它几乎不起作用，斜率应与去掉该点相差不超过 0.05%。
    """
    base = make_data(sigma_lo=0.002, sigma_hi=0.004, seed=709)
    dropped, ctx_a = run_case(_drop_point(base, 120.0))
    assert isinstance(dropped, dict), "去掉单点后没产出证书：stderr=%r" % (ctx_a["proc"].stderr or "")
    boosted, ctx_b = run_case(_boost_sigma(base, 120.0, 60.0))
    assert isinstance(boosted, dict), "放大单点不确定度后没产出证书：stderr=%r" % (ctx_b["proc"].stderr or "")
    a = float(dropped["expansion_terms"]["slope_um_per_c"])
    b = float(boosted["expansion_terms"]["slope_um_per_c"])
    assert abs(a - b) <= 0.0005 * abs(a), "大不确定度的点权重没压下去：%r 对 %r" % (b, a)


@pytest.mark.parametrize("case", ("sigma_zero", "few_points", "flat", "below_min_used"))
def test_invalid_data_rejected_without_output(case):
    """题面原文：「并在 stderr 说清缘由，不生成证书」。

    四种坏记录：sigma 非正、采样点不足十二、没有温度跨度、剔除后有效点太少。
    """
    data = _corrupt(make_data(seed=701), case)
    report, ctx = run_case(data)
    assert report is None, "坏记录 %s 仍产出了证书" % (case,)
    assert ctx["proc"].returncode == 2, "坏记录 %s 的退出码应为 2，实际 %r" % (case, ctx["proc"].returncode)
    assert (ctx["proc"].stderr or "").strip(), "坏记录 %s 没在 stderr 说清缘由" % (case,)
    assert not os.path.exists(ctx["out"]), "坏记录 %s 不该留下报告文件" % (case,)
    assert not [n for n in os.listdir(ctx["dir"]) if n.endswith(".tmp")], \
        "坏记录 %s 留下了临时文件" % (case,)


def test_report_schema_and_rounding():
    """题面原文：「证书里的每个数都必须是有限值」。

    校验顶层字段、位数与有限性，文件里不出现 NaN 或 Infinity。
    """
    data = make_data(seed=809)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    need = {"recorded_points", "fitted_points", "degrees_of_freedom", "reference_temperature",
            "expansion_terms", "transition_terms", "term_uncertainties",
            "transition_uncertainties", "slope_intercept_correlation", "residual_chi2",
            "error_correlation", "common_mode_sigma", "verdict", "checks", "certificate_note"}
    missing = sorted(need - set(report))
    assert not missing, "证书缺字段：%r" % (missing,)
    terms = report["expansion_terms"]
    uncs = report["term_uncertainties"]
    six = [float(terms["intercept_um"]), float(terms["slope_um_per_c"]),
           float(terms["curvature_um_per_c2"]), float(uncs["intercept_um"]),
           float(uncs["slope_um_per_c"]), float(uncs["curvature_um_per_c2"]),
           float(report["transition_terms"]["amplitude_um"]),
           float(report["transition_uncertainties"]["amplitude_um"])]
    assert all(math.isfinite(v) for v in six), "系数或标准误里有非有限值：%r" % (six,)
    assert all(abs(v - round(v, 6)) <= 1e-12 for v in six), "系数或标准误没保留六位小数：%r" % (six,)
    three = [float(report["transition_terms"]["midpoint_c"]),
             float(report["transition_terms"]["width_c"]),
             float(report["transition_uncertainties"]["midpoint_c"]),
             float(report["transition_uncertainties"]["width_c"])]
    assert all(abs(v - round(v, 3)) <= 1e-12 for v in three), "相变参数没保留三位小数：%r" % (three,)
    pair = [float(report["slope_intercept_correlation"]), float(report["residual_chi2"])]
    assert all(abs(v - round(v, 3)) <= 1e-12 for v in pair), "相关系数或卡方没保留三位：%r" % (pair,)
    assert int(report["degrees_of_freedom"]) == int(report["fitted_points"]) - 6, \
        "自由度不等于拟合点数减六：%r" % (report["degrees_of_freedom"],)
    assert set(report["certificate_note"]) >= {"method", "unit"}, \
        "certificate_note 缺 method 或 unit：%r" % (report["certificate_note"],)
    raw = open(ctx["out"], encoding="utf-8").read()
    assert "NaN" not in raw and "Infinity" not in raw, "证书里出现了 NaN 或 Infinity"


def test_rerun_produces_identical_bytes():
    """题面原文：「同一份台账跑两次，证书要一模一样，别把生成时间写进去」。

    连跑两次，两个证书文件字节完全相同。
    """
    data = make_data(seed=907)
    first, ctx1 = run_case(data)
    assert isinstance(first, dict), "第一次没产出证书：stderr=%r" % (ctx1["proc"].stderr or "")
    body1 = open(ctx1["out"], "rb").read()
    second, ctx2 = run_case(data)
    assert isinstance(second, dict), "第二次没产出证书：stderr=%r" % (ctx2["proc"].stderr or "")
    body2 = open(ctx2["out"], "rb").read()
    assert body1 == body2, "两次运行的证书不一致（里面带了时间戳？）"
    assert b"generated_at" not in body1, "证书里出现了生成时间字段"


def test_failed_run_keeps_previous_report():
    """题面原文：「失败的这次运行不能改写上一张证书，旧文件保持原样」。

    先跑一次合格记录留下证书，再用 sigma 非正的坏记录跑：退出码 2，旧证书字节不变。
    """
    good = make_data(seed=1009)
    tmp = tempfile.mkdtemp(prefix="te-keep-")
    dpath = os.path.join(tmp, "dilatometer.json")
    opath = os.path.join(tmp, "report.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False)
    first = _run(dpath, opath)
    assert first.returncode == 0, "合格记录跑失败了：%r" % (first.stderr or "")
    assert os.path.exists(opath), "合格记录没落证书"
    before = open(opath, "rb").read()
    broken = json.loads(json.dumps(good))
    broken["samples"][3]["sigma"] = -1.0
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    second = _run(dpath, opath)
    assert second.returncode == 2, "坏记录退出码应为 2，实际 %r" % (second.returncode,)
    assert (second.stderr or "").strip(), "坏记录没在 stderr 说清缘由"
    assert open(opath, "rb").read() == before, "失败的运行改写了上一张证书"


def test_zero_plateau_tolerance_rejected():
    """题面原文：「剔除后有效点少于」。

    把恒温容差设成 0（边界值）：必须退出码 2 且不产出证书。
    """
    data = make_data(seed=821)
    data["fit"]["plateau_tolerance_c"] = 0
    report, ctx = run_case(data)
    assert report is None, "恒温容差为 0 时仍产出了证书"
    assert ctx["proc"].returncode == 2, "恒温容差为 0 时退出码应为 2，实际 %r" % (ctx["proc"].returncode,)
    assert (ctx["proc"].stderr or "").strip(), "恒温容差为 0 时没在 stderr 说清缘由"
    assert not os.path.exists(ctx["out"]), "恒温容差为 0 时不该留下报告文件"


def test_verdict_accept_when_within_limits():
    """题面原文：「三项都通过写 ACCEPT」。

    数据与标称值一致、残差在限内、相变中点也对：verdict 必须是 ACCEPT，三项 check 都通过。
    """
    data = make_data(seed=1009, sigma_lo=0.002, sigma_hi=0.004)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    assert report["verdict"] == "ACCEPT", "本该判合格，实际 %r" % (report["verdict"],)
    checks = _checks(report)
    assert set(checks) == {"slope_vs_nominal", "residual_chi2", "midpoint_vs_nominal"}, \
        "checks 项不对：%r" % (sorted(checks),)
    assert all(item["passed"] for item in checks.values()), "ACCEPT 却有未通过的检查项"
    assert abs(float(checks["slope_vs_nominal"]["value"])) <= 1.0, \
        "合格件的斜率偏离超过容差：%r" % (checks["slope_vs_nominal"],)


def test_verdict_rejects_off_nominal_slope():
    """题面原文：「有一项不通过写 REJECT」。

    把标称斜率改成偏离 3% 的值（容差 1%）：必须判 REJECT，且 slope_vs_nominal 未通过。
    """
    data = make_data(seed=1013, sigma_lo=0.002, sigma_hi=0.004)
    nominal = data["inspection"]["slope_nominal_um_per_c"]
    data["inspection"]["slope_nominal_um_per_c"] = round(nominal * 1.03, 6)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    assert report["verdict"] == "REJECT", "斜率超差却判合格：%r" % (report["verdict"],)
    checks = _checks(report)
    assert checks["slope_vs_nominal"]["passed"] is False, "斜率超差却没标未通过"
    assert float(checks["slope_vs_nominal"]["value"]) > float(checks["slope_vs_nominal"]["limit"]), \
        "斜率偏离没超过容差：%r" % (checks["slope_vs_nominal"],)


def test_verdict_rejects_bad_residual_chi2():
    """题面原文：「有一项不通过写 REJECT」。

    把残差上限压到 0.2（实际约 1.0）：必须判 REJECT，且 residual_chi2 未通过。
    """
    data = make_data(seed=1019, sigma_lo=0.002, sigma_hi=0.004)
    data["inspection"]["max_residual_chi2"] = 0.2
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    assert report["verdict"] == "REJECT", "残差超限却判合格：%r" % (report["verdict"],)
    checks = _checks(report)
    assert checks["residual_chi2"]["passed"] is False, "残差超限却没标未通过"
    assert float(checks["residual_chi2"]["value"]) > 0.2, \
        "残差没超过压下来的上限：%r" % (checks["residual_chi2"],)


def test_verdict_rejects_off_nominal_midpoint():
    """题面原文：「的偏差（°C）超过」。

    把标称相变中点挪到偏离 8 °C 的位置（容差 3 °C）：必须判 REJECT，且 midpoint_vs_nominal 未通过。
    """
    data = make_data(seed=1021, sigma_lo=0.002, sigma_hi=0.004)
    data["inspection"]["midpoint_nominal_c"] = 188.0
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    assert report["verdict"] == "REJECT", "相变中点超差却判合格：%r" % (report["verdict"],)
    checks = _checks(report)
    assert checks["midpoint_vs_nominal"]["passed"] is False, "中点超差却没标未通过"
    assert float(checks["midpoint_vs_nominal"]["value"]) > 3.0, \
        "中点偏差没超过容差：%r" % (checks["midpoint_vs_nominal"],)
