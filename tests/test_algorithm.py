# -*- coding: utf-8 -*-
"""
基础自检：

    python tests/test_algorithm.py

覆盖三件事：
1. 常量 hazard 下 R[0] 恒等于 H（算法性质，不是 bug）；
2. 短运行长度质量能在变点处显著抬升；
3. 离线前向-后向给出的 Q(0) 有限，且后验概率归一在 [0, 1]。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bocpd import BOCPDConfig, BOCPDDetector  # noqa: E402
from src.bayesian_offline import BayesConfig, BayesianChangePointDetector  # noqa: E402


def _simulate(seed=11, n=300, cp=150):
    rng = np.random.default_rng(seed)
    mu = np.where(np.arange(n) < cp, 0.0, 0.006)
    return rng.normal(mu, 0.01)


def test_r0_equals_hazard():
    x = _simulate()
    det = BOCPDDetector(BOCPDConfig(hazard=1 / 50), log_callback=lambda *_: None)
    det.fit(x)
    assert np.allclose(det.changepoint_probs, det.config.hazard), "R[0] 应恒等于 hazard"


def test_short_run_mass_spikes_at_change():
    x = _simulate()
    det = BOCPDDetector(BOCPDConfig(hazard=1 / 50, short_run_k=8),
                        log_callback=lambda *_: None)
    cps = det.fit(x)
    s = det.short_run_probs
    assert s[140:175].max() > 0.5, "变点附近短运行长度质量应超过 0.5"
    assert len(cps) >= 1
    assert any(abs(c - 150) <= 15 for c in cps), f"应在 150 附近检出变点，实得 {cps}"


def test_offline_posterior_is_valid():
    rng = np.random.default_rng(5)
    n = 120
    x = rng.normal(0, 0.01, n)
    x[60:] = rng.normal(0.004, 0.02, n - 60)

    cfg = BayesConfig(p_max=1, lambda_param=1 / 40, k=200,
                      max_segment_length=60, min_gap=5, verbose=False)
    det = BayesianChangePointDetector(cfg, log_callback=lambda *_: None)
    cps = det.fit(x)

    p = det.posterior_probs
    assert p.shape == (n,)
    assert np.all((p >= 0) & (p <= 1))
    assert np.isfinite(det.log_Q_array[0])
    assert all(0 <= c < n for c in cps)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
