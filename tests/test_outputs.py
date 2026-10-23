#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判分输出：十九条行为测试按重要程度加权，权重和 = 1，另有一条门禁用例。

每条测试自己造一段升温记录，按加权最小二乘 / 广义最小二乘独立复算系数、标准误、
相关系数与卡方，再驱动被测程序的命令行入口比对证书。期望值不读被测代码，
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
        y = a0 + a1 * dT + a2 * dT * dT + noise
        samples.append({"t": temp, "L": round(y, 6), "sigma": round(sigma, 6)})
    return {"unit": {"temperature": "degC", "length": "um"},
            "reference_temperature": t_ref, "samples": samples,
            "inspection": {"slope_nominal_um_per_c": a1, "slope_tolerance_pct": 1.0,
                           "max_residual_chi2": 1.5},
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
        out["fit"]["min_used"] = 250
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

    出厂记录：三个系数都要与独立复算的加权拟合一致，相对差不超过 0.5%。
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


def test_std_errors_reflect_point_weights():
    """题面原文：「各点不确定度不同，不能当等权处理」。

    逐点不确定度从 0.002 涨到 0.05 的记录：a1 的标准误必须与加权复算一致（5% 以内）。
    把 sigma 当等权处理时，标准误会差出好几倍。
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

    换成 40→400 °C、a1=1.6、a2=0.004 的记录后仍要吻合。
    """
    data = make_data(a1=1.6, a2=0.004, t_start=40.0, t_end=400.0, step=2.0,
                     plateau_at=200.0, plateau_len=20, sigma_lo=0.004, sigma_hi=0.02,
                     seed=409)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    exp = expect(data)
    terms = report["expansion_terms"]
    slope = float(terms["slope_um_per_c"])
    curv = float(terms["curvature_um_per_c2"])
    assert abs(slope - exp["beta"][1]) <= 0.0015 * abs(exp["beta"][1]), \
        "换量程后膨胀系数没对上：被测 %r，独立复算 %r" % (slope, exp["beta"][1])
    assert abs(curv - exp["beta"][2]) <= 0.0015 * abs(exp["beta"][2]), \
        "换量程后二次项没对上：被测 %r，独立复算 %r" % (curv, exp["beta"][2])
    assert int(report["degrees_of_freedom"]) == exp["degrees_of_freedom"], \
        "自由度不对：被测 %r，应为 %r" % (report["degrees_of_freedom"], exp["degrees_of_freedom"])


def test_chi2_reduced_near_unity():
    """题面原文：「chi2_red = 白化后残差平方和 /(n - 3)」。

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
    把相关误差当成独立误差（普通加权最小二乘）会明显偏。
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

    叠上 τ=0.0025 的共模误差：斜率要落在独立复算的 0.3% 以内，标准误在 5% 以内，
    且报告回填了 common_mode_sigma。漏掉共模项的实现，斜率与标准误都会被低估。
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
    data = make_data(t_start=25.0, t_end=45.0, step=1.0, plateau_at=35.0, plateau_len=6, seed=701)
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
    data = make_data(t_start=25.0, t_end=45.0, step=1.0, plateau_at=35.0, plateau_len=6, seed=703)
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
            "expansion_terms", "term_uncertainties", "slope_intercept_correlation",
            "residual_chi2", "error_correlation", "common_mode_sigma", "verdict", "checks",
            "certificate_note"}
    missing = sorted(need - set(report))
    assert not missing, "证书缺字段：%r" % (missing,)
    terms = report["expansion_terms"]
    uncs = report["term_uncertainties"]
    six = [float(terms["intercept_um"]), float(terms["slope_um_per_c"]),
           float(terms["curvature_um_per_c2"]), float(uncs["intercept_um"]),
           float(uncs["slope_um_per_c"]), float(uncs["curvature_um_per_c2"])]
    assert all(math.isfinite(v) for v in six), "系数或标准误里有非有限值：%r" % (six,)
    assert all(abs(v - round(v, 6)) <= 1e-12 for v in six), "系数或标准误没保留六位小数：%r" % (six,)
    three = [float(report["slope_intercept_correlation"]), float(report["residual_chi2"])]
    assert all(abs(v - round(v, 3)) <= 1e-12 for v in three), "相关系数或卡方没保留三位：%r" % (three,)
    assert int(report["degrees_of_freedom"]) == int(report["fitted_points"]) - 3, \
        "自由度不等于拟合点数减三：%r" % (report["degrees_of_freedom"],)
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
    """题面原文：「两项都通过写 ACCEPT」。

    数据与标称值一致、残差在限内：verdict 必须是 ACCEPT，且两项 check 都通过。
    """
    data = make_data(sigma_lo=0.002, sigma_hi=0.004, seed=1009)
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    assert report["verdict"] == "ACCEPT", "本该判合格，实际 %r" % (report["verdict"],)
    checks = _checks(report)
    assert set(checks) == {"slope_vs_nominal", "residual_chi2"}, "checks 项不对：%r" % (sorted(checks),)
    assert all(item["passed"] for item in checks.values()), "ACCEPT 却有未通过的检查项"
    assert abs(float(checks["slope_vs_nominal"]["value"])) <= 1.0, \
        "合格件的斜率偏离超过容差：%r" % (checks["slope_vs_nominal"],)


def test_verdict_rejects_off_nominal_slope():
    """题面原文：「有一项不通过写 REJECT」。

    把标称斜率改成偏离 3% 的值（容差 1%）：必须判 REJECT，且 slope_vs_nominal 未通过。
    """
    data = make_data(sigma_lo=0.002, sigma_hi=0.004, seed=1013)
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

    把残差上限压到 0.5（实际约 1.0）：必须判 REJECT，且 residual_chi2 未通过。
    """
    data = make_data(sigma_lo=0.002, sigma_hi=0.004, seed=1019)
    data["inspection"]["max_residual_chi2"] = 0.5
    report, ctx = run_case(data)
    assert isinstance(report, dict), "没产出证书：stderr=%r" % (ctx["proc"].stderr or "")
    assert report["verdict"] == "REJECT", "残差超限却判合格：%r" % (report["verdict"],)
    checks = _checks(report)
    assert checks["residual_chi2"]["passed"] is False, "残差超限却没标未通过"
    assert float(checks["residual_chi2"]["value"]) > 0.5, \
        "残差没超过压下来的上限：%r" % (checks["residual_chi2"],)
