#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""膨胀仪升温曲线复核：加权最小二乘拟合二次膨胀式与参数不确定度。

命令行：
    python3 src/expansion.py --data data/dilatometer.json --out out/report.json

口径：
  · 模型 L = a0 + a1·ΔT + a2·ΔT²，ΔT = T - reference_temperature；
  · 逐点不确定度 sigma 来自仪器，权重取 1/sigma²（异方差时不能当等权处理）；
  · 恒温保持段：温度与上一点相差小于 plateau_tolerance_c 的采样视为保持点，
    每组连续保持点只保留第一个（其余不带来新信息，还会把权重堆到同一温度上）；
  · 参数协方差 = sigma_hat²·(XᵀWX)⁻¹，sigma_hat² = 加权残差平方和 /(n - 3)；
  · chi2_red = 加权残差平方和 /(n - 3)；
  · a1、a2 的相关系数取协方差矩阵对应元素归一化；
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
    tol = cfg.get("plateau_tolerance_c", 0.02)
    min_used = cfg.get("min_used", 10)
    if not isinstance(tol, (int, float)) or isinstance(tol, bool) or float(tol) <= 0:
        _fail("fit.plateau_tolerance_c 要是正数")
    if not isinstance(min_used, int) or isinstance(min_used, bool) or min_used < 6:
        _fail("fit.min_used 要是不小于 6 的整数")

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
            "min_used": int(min_used), "unit": (raw.get("unit") or {})}


def keep_plateau_free(rows, tol):
    """每组连续恒温保持点只留第一个。"""
    kept = []
    for r in rows:
        if kept and abs(r["t"] - kept[-1]["t"]) < tol:
            continue
        kept.append(r)
    return kept


# ---------------------------------------------------------------- 线性代数（3x3）
def solve3(a, b):
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-14:
            _fail("设计矩阵奇异，拟合窗口里温度跨度不足")
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for k in range(col, 4):
            m[col][k] /= pv
        for r in range(3):
            if r == col:
                continue
            f = m[r][col]
            if f:
                for k in range(col, 4):
                    m[r][k] -= f * m[col][k]
    return [m[0][3], m[1][3], m[2][3]]


def inverse3(a):
    inv = [[0.0] * 3 for _ in range(3)]
    for j in range(3):
        e = [1.0 if i == j else 0.0 for i in range(3)]
        col = solve3(a, e)
        for i in range(3):
            inv[i][j] = col[i]
    return inv


def analyse(scan):
    rows = keep_plateau_free(scan["rows"], scan["tol"])
    if len(rows) < scan["min_used"]:
        _fail("剔除恒温保持段后只剩 %d 个有效点，少于要求的 %d" % (len(rows), scan["min_used"]))
    span = max(r["t"] for r in rows) - min(r["t"] for r in rows)
    if span <= 0:
        _fail("有效点没有温度跨度")

    xtwx = [[0.0] * 3 for _ in range(3)]
    xtwy = [0.0] * 3
    for r in rows:
        dT = r["t"] - scan["t_ref"]
        x = [1.0, dT, dT * dT]
        w = 1.0 / (r["sigma"] * r["sigma"])
        for i in range(3):
            xtwy[i] += w * x[i] * r["y"]
            for j in range(3):
                xtwx[i][j] += w * x[i] * x[j]
    beta = solve3(xtwx, xtwy)

    sse = 0.0
    for r in rows:
        dT = r["t"] - scan["t_ref"]
        model = beta[0] + beta[1] * dT + beta[2] * dT * dT
        w = 1.0 / (r["sigma"] * r["sigma"])
        sse += w * (r["y"] - model) ** 2

    dof = len(rows) - 3
    var = sse / dof
    cov = [[var * v for v in row] for row in inverse3(xtwx)]
    se = [math.sqrt(max(cov[i][i], 0.0)) for i in range(3)]
    denom = math.sqrt(cov[1][1] * cov[2][2])
    corr = cov[1][2] / denom if denom > 0 else 0.0
    chi2 = sse / dof
    return {"beta": beta, "se": se, "corr": corr, "chi2": chi2,
            "n_total": len(scan["rows"]), "n_used": len(rows), "dof": dof}


def build_report(scan, res):
    return {
        "n_total": res["n_total"],
        "n_used": res["n_used"],
        "dof": res["dof"],
        "reference_temperature": scan["t_ref"],
        "coefficients": {"a0": round(res["beta"][0], 6),
                         "a1": round(res["beta"][1], 6),
                         "a2": round(res["beta"][2], 6)},
        "std_errors": {"a0": round(res["se"][0], 6),
                       "a1": round(res["se"][1], 6),
                       "a2": round(res["se"][2], 6)},
        "corr_a1_a2": round(res["corr"], 3),
        "chi2_red": round(res["chi2"], 3),
        "meta": {"method": "wls-quadratic", "unit": scan["unit"]},
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
