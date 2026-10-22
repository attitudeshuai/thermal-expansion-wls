#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""膨胀仪升温曲线复核：拟合二次膨胀式并给出参数。

用法：
    python3 src/expansion.py --data data/dilatometer.json --out out/report.json
"""
import argparse
import datetime as dt
import json
import math
import os


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def solve3(a, b):
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        pv = m[col][col]
        for k in range(col, 4):
            m[col][k] /= pv
        for r in range(3):
            if r != col:
                f = m[r][col]
                for k in range(col, 4):
                    m[r][k] -= f * m[col][k]
    return [m[0][3], m[1][3], m[2][3]]


def analyse(raw):
    t_ref = float(raw["reference_temperature"])
    rows = raw["samples"]

    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for r in rows:
        dT = float(r["t"]) - t_ref
        x = [1.0, dT, dT * dT]
        y = float(r["L"])
        for i in range(3):
            xty[i] += x[i] * y
            for j in range(3):
                xtx[i][j] += x[i] * x[j]
    beta = solve3(xtx, xty)

    sse = 0.0
    for r in rows:
        dT = float(r["t"]) - t_ref
        model = beta[0] + beta[1] * dT + beta[2] * dT * dT
        sse += (float(r["L"]) - model) ** 2
    rmse = math.sqrt(sse / len(rows))

    return {
        "n_total": len(rows),
        "reference_temperature": t_ref,
        "coefficients": {"a0": round(beta[0], 6), "a1": round(beta[1], 6), "a2": round(beta[2], 6)},
        "rmse": round(rmse, 6),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": {"method": "ols-quadratic"},
    }


def main():
    ap = argparse.ArgumentParser(description="膨胀仪曲线拟合")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = load(args.data)
    report = analyse(raw)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
