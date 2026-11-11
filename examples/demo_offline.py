# -*- coding: utf-8 -*-
"""
离线贝叶斯多变点演示：AR(2) 结构变化 + 精确后验

运行：
    python examples/demo_offline.py

输出：
    examples/figures/offline_demo.png
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bayesian_offline import BayesConfig, BayesianChangePointDetector  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def simulate(n=300, seed=3):
    """AR(1) 系数在 100 / 200 处切换（持续性 0 → 0.9 → -0.5）。"""
    rng = np.random.default_rng(seed)
    true_cps = [100, 200]
    phi = np.array([0.0, 0.9, -0.5])
    sigma = np.array([0.008, 0.018, 0.012])

    x = np.zeros(n)
    for t in range(1, n):
        seg = 0 if t < true_cps[0] else (1 if t < true_cps[1] else 2)
        x[t] = phi[seg] * x[t - 1] + rng.normal(0, sigma[seg])
    return x, true_cps


def main():
    x, true_cps = simulate()

    # p_c 取 0.35：变点位置在真实边界附近几期内分散，单个位置的后验峰通常到不了 0.5
    cfg = BayesConfig(p_max=2, lambda_param=1 / 40, p_c=0.35, k=1500,
                      max_segment_length=120, min_gap=8, verbose=True)
    det = BayesianChangePointDetector(cfg, log_callback=lambda *_: None)

    t0 = time.time()
    cps = det.fit(x)
    elapsed = time.time() - t0

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.4]},
    )

    ax = axes[0]
    ax.plot(x, color="#1f3a5f", lw=1.2, label="合成收益率序列")
    for i, cp in enumerate(true_cps):
        ax.axvline(cp, color="#8c8c8c", ls=":", lw=1.6,
                   label="真实变点" if i == 0 else None)
    for i, cp in enumerate(cps):
        ax.axvline(cp, color="#d62728", ls="--", lw=1.3,
                   label="后验检测变点" if i == 0 else None)
    ax.set_title("离线贝叶斯多变点估计（AR(p) 分段边际似然 + 精确后验采样）",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("收益率")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(det.posterior_probs, color="#6a3d9a", lw=1.2,
            label=r"后验包含概率 $P(\tau = t \mid x_{1:n})$")
    ax.axhline(cfg.p_c, color="#ff7f00", ls="--", lw=1.2, label=f"阈值 {cfg.p_c}")
    for i, cp in enumerate(true_cps):
        ax.axvline(cp, color="#8c8c8c", ls=":", lw=1.6)
    ax.set_xlabel("期数（周）")
    ax.set_ylabel("后验概率")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "offline_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")

    print(f"图片已保存: {out}")
    print(f"耗时: {elapsed:.1f}s")
    print(f"真实变点: {true_cps}")
    print(f"检测变点: {cps}")


if __name__ == "__main__":
    main()
