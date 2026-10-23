#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""膨胀仪升温曲线复核：加权二次 + 相变台阶的广义最小二乘，含参数不确定度与首件判定。

命令行：
    python3 src/expansion.py --data data/dilatometer.json --out out/report.json

口径：
  · 模型 L(ΔT) = a0 + a1·ΔT + a2·ΔT² + A·S((ΔT − Tc)/w)，ΔT = t − reference_temperature，
    S(z) = 1/(1+exp(−z)) 是相变台阶（Tc 中点、w 宽度、A 台阶高度）；
  · 给定 Tc 与 w 时 [a0, a1, a2, A] 是线性的，所以先在 (Tc, w) 上粗定位、再逐个细化；
  · 协方差矩阵是 Σ = D·R·D + τ²·J：D 为逐点标准差、R_ij = ρ^|i-j|、τ 是共模项
    （fit.common_mode_sigma）；ρ 或 τ 非零时整体 Cholesky 白化，再在白化空间里解正规方程；
  · 恒温保持段：温度与上一点相差小于 plateau_tolerance_c 的采样视为保持点，每组只留第一个；
  · 参数协方差 = sigma_hat²·(JᵀJ)⁻¹（白化空间的 6 列雅可比，含 Tc、w 两列），
    sigma_hat² = 白化残差平方和 /(n − 6)；标准误取对角线开方；
  · chi2_red = 白化残差平方和 /(n − 6)，error_correlation 与 common_mode_sigma 原样回填；
  · 首件判定按 inspection 段三项：斜率相对标称值的偏离百分比、chi2_red、相变中点偏差；
  · 输出先写临时文件再原子替换；非法输入以退出码 2 结束且不落文件。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

EXIT_INVALID = 2
NPAR = 6


def _fail(msg):
    sys.stderr.write("非法输入：%s\n" % msg)
    raise SystemExit(EXIT_INVALID)


def load_data(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        _fail("读不到试验记录 %s" % path)
    except ValueError as e:
        _fail("记录不是合法 JSON：%s" % e)
    if not isinstance(raw, dict):
        _fail("记录顶层要是对象")
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) < 12:
        _fail("samples 至少要 12 个采样点")
    t_ref = raw.get("reference_temperature")
    if not isinstance(t_ref, (int, float)) or isinstance(t_ref, bool):
        _fail("reference_temperature 要是数值")
    cfg = raw.get("fit") or {}
    rho = cfg.get("error_correlation", 0.0)
    if not isinstance(rho, (int, float)) or isinstance(rho, bool) or not (float(rho) >= 0.0 and float(rho) < 0.95):
        _fail("fit.error_correlation 要落在 [0, 0.95)")
    rho = float(rho)
    common = cfg.get("common_mode_sigma", 0.0)
    if not isinstance(common, (int, float)) or isinstance(common, bool) or float(common) < 0:
        _fail("fit.common_mode_sigma 要是不小于 0 的数")
    common = float(common)
    tol = cfg.get("plateau_tolerance_c", 0.02)
    min_used = cfg.get("min_used", 10)
    if not isinstance(tol, (int, float)) or isinstance(tol, bool) or float(tol) <= 0:
        _fail("fit.plateau_tolerance_c 要是正数")
    if not isinstance(min_used, int) or isinstance(min_used, bool) or min_used < 12:
        _fail("fit.min_used 要是不小于 12 的整数")

    insp = raw.get("inspection")
    if not isinstance(insp, dict):
        _fail("缺 inspection 判定规则段")
    slope_nominal = insp.get("slope_nominal_um_per_c")
    slope_tol = insp.get("slope_tolerance_pct", 1.0)
    chi2_limit = insp.get("max_residual_chi2", 1.5)
    mid_nominal = insp.get("midpoint_nominal_c")
    mid_tol = insp.get("midpoint_tolerance_c", 3.0)
    for name, val in (("slope_nominal_um_per_c", slope_nominal),
                      ("slope_tolerance_pct", slope_tol),
                      ("max_residual_chi2", chi2_limit),
                      ("midpoint_nominal_c", mid_nominal),
                      ("midpoint_tolerance_c", mid_tol)):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(float(val)):
            _fail("inspection.%s 要是有限数值" % name)
    if float(slope_tol) <= 0 or float(chi2_limit) <= 0 or float(mid_tol) <= 0:
        _fail("inspection 的容差与上限都要为正")
    rows = []
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            _fail("samples[%d] 不是对象" % i)
        try:
            t = float(s["t"])
            y = float(s["L"])
            sig = float(s["sigma"])
        except (KeyError, TypeError, ValueError):
            _fail("samples[%d] 的 t/L/sigma 必须是数值" % i)
        if not (math.isfinite(t) and math.isfinite(y) and math.isfinite(sig)):
            _fail("samples[%d] 有非有限数值" % i)
        if sig <= 0:
            _fail("samples[%d].sigma 必须为正" % i)
        rows.append({"t": t, "y": y, "sigma": sig})
    return {"rows": rows, "t_ref": float(t_ref), "tol": float(tol),
            "min_used": int(min_used), "unit": (raw.get("unit") or {}), "rho": rho,
            "common": common, "slope_nominal": float(slope_nominal),
            "slope_tol": float(slope_tol), "chi2_limit": float(chi2_limit),
            "mid_nominal": float(mid_nominal), "mid_tol": float(mid_tol)}


def keep_plateau_free(rows, tol):
    """每组连续恒温保持点只留第一个。"""
    kept = []
    for r in rows:
        if kept and abs(r["t"] - kept[-1]["t"]) < tol:
            continue
        kept.append(r)
    return kept


def logistic(z):
    if z >= 0.0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _cholesky(mat):
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


def _forward_sub(low, vec):
    n = len(vec)
    out = [0.0] * n
    for i in range(n):
        acc = vec[i] - sum(low[i][k] * out[k] for k in range(i))
        out[i] = acc / low[i][i]
    return out


def _build_whitener(sigmas, rho, common):
    """返回一个把列向量白化的函数（Σ = D·R·D + τ²·J）。"""
    n = len(sigmas)
    if rho <= 0.0 and common <= 0.0:
        return lambda vec: [vec[i] / sigmas[i] for i in range(n)]
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cov[i][j] = sigmas[i] * sigmas[j] * (rho ** abs(i - j)) + common * common
    low = _cholesky(cov)
    if low is None:
        return None

    def whiten(vec):
        return _forward_sub(low, list(vec))

    return whiten


def solve_linear(a, b):
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


def inverse_n(a):
    n = len(a)
    inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        col = solve_linear(a, [1.0 if i == j else 0.0 for i in range(n)])
        if col is None:
            return None
        for i in range(n):
            inv[i][j] = col[i]
    return inv


def _normal_equations(cols, y):
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


def analyse(scan):
    rows = keep_plateau_free(scan["rows"], scan["tol"])
    if len(rows) < scan["min_used"]:
        _fail("剔除恒温保持段后只剩 %d 个有效点，少于要求的 %d" % (len(rows), scan["min_used"]))
    span = max(r["t"] for r in rows) - min(r["t"] for r in rows)
    if span <= 0:
        _fail("有效点没有温度跨度")

    t_ref = scan["t_ref"]
    dt = [r["t"] - t_ref for r in rows]
    ts = [r["t"] for r in rows]
    ys = [r["y"] for r in rows]
    sigmas = [r["sigma"] for r in rows]
    whiten = _build_whitener(sigmas, scan["rho"], scan["common"])
    if whiten is None:
        _fail("协方差矩阵不正定，检查 fit.error_correlation / fit.common_mode_sigma")

    wx0 = whiten([1.0] * len(rows))
    wx1 = whiten(dt)
    wx2 = whiten([v * v for v in dt])
    wy = whiten(ys)
    base_cols = [wx0, wx1, wx2]

    def sse_for(tc, width):
        step = whiten([logistic((v - tc) / width) for v in ts])
        cols = base_cols + [step]
        xtx, xty = _normal_equations(cols, wy)
        coef = solve_linear(xtx, xty)
        if coef is None:
            return None, None
        sse = 0.0
        for i in range(len(wy)):
            model = coef[0] * wx0[i] + coef[1] * wx1[i] + coef[2] * wx2[i] + coef[3] * step[i]
            d = wy[i] - model
            sse += d * d
        return coef, sse

    lo_t = min(r["t"] for r in rows)
    hi_t = max(r["t"] for r in rows)
    best = None
    tc_grid = [lo_t + (hi_t - lo_t) * i / 24.0 for i in range(25)]
    width_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0, 15.0]
    for tc in tc_grid:
        for width in width_grid:
            coef, sse = sse_for(tc, width)
            if sse is None:
                continue
            if best is None or sse < best[0]:
                best = (sse, tc, width)
    if best is None:
        _fail("相变台阶拟合没找到可行解")

    def golden(fn, lo, hi, iters=60):
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

    sse_best, tc_best, w_best = best
    for _ in range(3):
        def f_tc(x):
            _, s = sse_for(x, w_best)
            return float("inf") if s is None else s

        def f_w(x):
            _, s = sse_for(tc_best, x)
            return float("inf") if s is None else s

        tc_best = golden(f_tc, tc_best - 6.0, tc_best + 6.0)
        w_hi = max(1.0, w_best * 2.0)
        w_lo = max(0.2, w_best / 2.0)
        w_best = golden(f_w, w_lo, w_hi)
        _, sse_best = sse_for(tc_best, w_best)

    coef, sse = sse_for(tc_best, w_best)
    if coef is None:
        _fail("最终拟合解不出来，检查记录")

    n = len(rows)
    dof = n - NPAR
    if dof <= 0:
        _fail("有效点太少，自由度不够")
    var = sse / dof

    step = whiten([logistic((v - tc_best) / w_best) for v in ts])
    j_cols = [wx0, wx1, wx2, step]
    dstep_dtc = [(-logistic((v - tc_best) / w_best) * (1.0 - logistic((v - tc_best) / w_best)) / w_best)
                 for v in ts]
    dstep_dw = [(-logistic((v - tc_best) / w_best) * (1.0 - logistic((v - tc_best) / w_best))
                 * (v - tc_best) / (w_best * w_best)) for v in ts]
    j_cols.append(whiten([coef[3] * v for v in dstep_dtc]))
    j_cols.append(whiten([coef[3] * v for v in dstep_dw]))
    xtx, _ = _normal_equations(j_cols, wy)
    inv = inverse_n(xtx)
    if inv is None:
        _fail("雅可比矩阵奇异，无法给出参数不确定度")
    cov = [[var * inv[i][j] for j in range(NPAR)] for i in range(NPAR)]
    se = [math.sqrt(max(cov[i][i], 0.0)) for i in range(NPAR)]
    denom = math.sqrt(cov[1][1] * cov[2][2])
    corr = cov[1][2] / denom if denom > 0 else 0.0

    return {"beta": coef[:4], "tc": tc_best, "width": w_best, "se": se, "corr": corr,
            "chi2": sse / dof, "recorded_points": len(scan["rows"]), "fitted_points": n,
            "degrees_of_freedom": dof}


def evaluate(scan, res):
    """按 inspection 里的规则逐项判定，返回 (verdict, checks)。"""
    slope_nominal = scan["slope_nominal"]
    slope_tol = scan["slope_tol"]
    chi2_limit = scan["chi2_limit"]
    slope = res["beta"][1]
    dev_pct = abs(slope - slope_nominal) / slope_nominal * 100.0
    chi2 = res["chi2"]
    mid_dev = abs(res["tc"] - scan["mid_nominal"])
    checks = [
        {"name": "slope_vs_nominal", "value": round(dev_pct, 4),
         "limit": round(slope_tol, 4), "passed": dev_pct <= slope_tol},
        {"name": "residual_chi2", "value": round(chi2, 4),
         "limit": round(chi2_limit, 4), "passed": chi2 <= chi2_limit},
        {"name": "midpoint_vs_nominal", "value": round(mid_dev, 4),
         "limit": round(scan["mid_tol"], 4), "passed": mid_dev <= scan["mid_tol"]},
    ]
    verdict = "ACCEPT" if all(item["passed"] for item in checks) else "REJECT"
    return verdict, checks


def build_report(scan, res):
    verdict, checks = evaluate(scan, res)
    return {
        "recorded_points": res["recorded_points"],
        "fitted_points": res["fitted_points"],
        "degrees_of_freedom": res["degrees_of_freedom"],
        "reference_temperature": scan["t_ref"],
        "expansion_terms": {"intercept_um": round(res["beta"][0], 6),
                            "slope_um_per_c": round(res["beta"][1], 6),
                            "curvature_um_per_c2": round(res["beta"][2], 6)},
        "transition_terms": {"midpoint_c": round(res["tc"], 3),
                             "width_c": round(res["width"], 3),
                             "amplitude_um": round(res["beta"][3], 6)},
        "term_uncertainties": {"intercept_um": round(res["se"][0], 6),
                               "slope_um_per_c": round(res["se"][1], 6),
                               "curvature_um_per_c2": round(res["se"][2], 6)},
        "transition_uncertainties": {"midpoint_c": round(res["se"][4], 3),
                                     "width_c": round(res["se"][5], 3),
                                     "amplitude_um": round(res["se"][3], 6)},
        "slope_intercept_correlation": round(res["corr"], 3),
        "residual_chi2": round(res["chi2"], 3),
        "error_correlation": round(scan["rho"], 3),
        "common_mode_sigma": round(scan["common"], 6),
        "verdict": verdict,
        "checks": checks,
        "certificate_note": {
            "method": "gls-transition" if (scan["rho"] > 0 or scan["common"] > 0) else "wls-transition",
            "unit": scan["unit"]},
    }


def atomic_write(path, obj):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    body = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".exp-", suffix=".tmp", dir=out_dir or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="膨胀仪升温曲线加权拟合")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    scan = load_data(args.data)
    res = analyse(scan)
    atomic_write(args.out, build_report(scan, res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
