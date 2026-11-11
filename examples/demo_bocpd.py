# -*- coding: utf-8 -*-
"""
BOCPD 演示：合成周线收益率上的在线变点检测。

运行：
    python examples/demo_bocpd.py

输出：
    examples/figures/bocpd_demo.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bocpd import BOCPDConfig, BOCPDDetector  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def simulate(n=520, seed=7):
    """四段式合成收益率：均值与波动率同时发生结构变化。"""
    rng = np.random.default_rng(seed)
    true_cps = [120, 260, 400]          # 收益率序列上的真实变点
    regime = np.zeros(n, dtype=int)
    regime[120:260] = 1
    regime[260:400] = 2
    regime[400:] = 3

    mu = np.array([0.004, -0.006, 0.008, -0.002])[regime]
    sigma = np.array([0.012, 0.025, 0.014, 0.020])[regime]
    return rng.normal(mu, sigma), true_cps


def main():
    returns, true_cps = simulate()

    cfg = BOCPDConfig(hazard=1 / 50, cp_threshold=0.5, min_gap=8,
                      decision_rule="run_length", short_run_k=8)
    det = BOCPDDetector(cfg, log_callback=lambda *_: None)
    dcps = det.fit(returns)

    price = 100.0 * np.exp(np.cumsum(returns))
    t = np.arange(len(returns))
    hit = [c for c in dcps if any(abs(c - t0) <= 15 for t0 in true_cps)]

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2, 1.2, 1.2]},
    )

    # 1) 价格 + 变点
    ax = axes[0]
    ax.plot(t, price, color="#1f3a5f", lw=1.3)
    for i, cp in enumerate(true_cps):
        ax.axvline(cp, color="#8c8c8c", ls=":", lw=1.5,
                   label="真实变点" if i == 0 else None)
    for i, cp in enumerate(dcps):
        ax.axvline(cp, color="#d62728", ls="--", lw=1.3,
                   label="检测变点" if i == 0 else None)
    ax.set_title("BOCPD 在线变点检测（合成周线数据）", fontsize=13, fontweight="bold")
    ax.set_ylabel("价格")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # 2) R[0]：常量 hazard 下的退化现象
    ax = axes[1]
    ax.plot(t, det.changepoint_probs, color="#6a3d9a", lw=1.2)
    ax.axhline(cfg.hazard, color="#6a3d9a", ls=":", lw=1.0)
    ax.set_ylim(-0.02, 0.25)
    ax.set_ylabel(r"$R_0(t)$")
    ax.set_title(r"$R_0(t)$ 恒等于 hazard $H=0.02$：常量 hazard 下无判别力",
                 fontsize=10, loc="left")
    ax.grid(alpha=0.3)

    # 3) 短运行长度后验质量：真正的判据
    ax = axes[2]
    ax.plot(t, det.short_run_probs, color="#1b9e77", lw=1.3,
            label=rf"$P(r_t \leq {cfg.short_run_k} \mid x_{{1:t}})$")
    ax.axhline(cfg.cp_threshold, color="#ff7f00", ls="--", lw=1.2, label="阈值 0.5")
    ax.set_ylabel("后验概率")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("「当前段刚刚重启」的概率", fontsize=10, loc="left")
    ax.grid(alpha=0.3)

    # 4) MAP 运行长度
    ax = axes[3]
    ax.plot(t, det.map_run_lengths, color="#e6821a", lw=1.2)
    ax.set_xlabel("期数（周）")
    ax.set_ylabel("MAP 运行长度")
    ax.set_title("MAP 运行长度：变点处归零重开", fontsize=10, loc="left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "bocpd_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")

    print(f"图片已保存: {out}")
    print(f"真实变点: {true_cps}")
    print(f"检测变点: {dcps}")
    print(f"其中命中真实变点（±15 期）: {hit}")


if __name__ == "__main__":
    main()
