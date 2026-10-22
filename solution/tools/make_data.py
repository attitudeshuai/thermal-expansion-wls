#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成膨胀仪升温记录（二次多项式 + 逐点不确定度 + 恒温保持段）。

数据口径：业务模型 + 确定性生成规则（固定伪随机序列，不用真随机）——
  L = a0 + a1·(T - T_ref) + a2·(T - T_ref)²，逐点叠加按 sigma 缩放的伪随机误差；
  sigma 可以是常数，也可以随温度变化（模拟高温端噪声变大）；
  中间有一段控温平台（温度重复），需要按规则剔除。
"""
import json
import math
import sys


class Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_data(a1=0.600, a2=0.0008, a0=0.0, t_ref=25.0, t_start=25.0, t_end=225.0,
              step=1.0, plateau_at=120.0, plateau_len=40, sigma_lo=0.003, sigma_hi=0.003,
              seed=20261025):
    rng = Lcg(seed)
    temps = []
    t = t_start
    while t <= t_end + 1e-9:
        temps.append(round(t, 2))
        t += step
    plateau = [round(plateau_at, 2)] * plateau_len
    pos = min(range(len(temps)), key=lambda k: abs(temps[k] - plateau_at))
    temps = temps[:pos] + plateau + temps[pos:]

    span = max(temps) - min(temps)
    samples = []
    for i, temp in enumerate(temps):
        frac = (temp - min(temps)) / span if span else 0.0
        sigma = sigma_lo + (sigma_hi - sigma_lo) * frac
        dT = temp - t_ref
        noise = (rng.next_u() + rng.next_u() - 1.0) * math.sqrt(6.0) * sigma   # std 恰为 sigma
        y = a0 + a1 * dT + a2 * dT * dT + noise
        samples.append({"t": temp, "L": round(y, 6), "sigma": round(sigma, 6)})

    return {
        "unit": {"temperature": "degC", "length": "um"},
        "reference_temperature": t_ref,
        "samples": samples,
        "fit": {"plateau_tolerance_c": 0.02, "min_used": 10},
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "dilatometer.json"
    data = make_data()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print("wrote %s  samples=%d" % (out, len(data["samples"])))
