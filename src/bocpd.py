# -*- coding: utf-8 -*-
"""
BOCPD —— Bayesian Online Changepoint Detection（在线贝叶斯变点检测）

核心递归（Adams & MacKay, 2007）：

    R_{t,r}  ∝  R_{t-1,r-1} · (1 - H) · p(x_t | r_{t-1}=r-1)
    R_{t,0}  ∝  Σ_r R_{t-1,r} · H · p(x_t | r_{t-1}=r)

其中 H 为 hazard rate（每期发生变点的先验概率），观测模型取
Normal-Inverse-Gamma 共轭先验，预测分布为 Student-t。

本模块只依赖 numpy / scipy，不包含任何 GUI 或数据获取逻辑，
可直接嵌入实盘系统或 notebook。
"""

from math import log

import numpy as np
from scipy.special import gammaln, logsumexp

__all__ = ["BOCPDConfig", "NIGState", "NIGPrior", "BOCPDDetector"]


class BOCPDConfig:
    """BOCPD 的超参数容器（与原 notebook 中的 Config 字段保持一致）。

    新增 ``decision_rule`` / ``short_run_k``：

    - ``decision_rule="run_length"``（默认）：用 P(r_t ≤ k | x_{1:t}) 判断变点，
      即「当前段已经重启过」的后验概率。
    - ``decision_rule="r0"``：沿用 notebook 的 R[0] 阈值法。

    注意：常量 hazard 下数学上恒有 R[0] = H（与数据无关），见 README「注意事项」。
    """

    def __init__(self, hazard=1.0 / 50.0, cp_threshold=0.5, min_gap=8,
                 max_run_length=5000, nig_mu0=0.0, nig_kappa0=1.0,
                 nig_alpha0=1.0, nig_beta0=1e-4, decision_rule="run_length",
                 short_run_k=8, verbose=True):
        self.hazard = hazard
        self.cp_threshold = cp_threshold
        self.min_gap = min_gap
        self.max_run_length = max_run_length
        self.nig_mu0 = nig_mu0
        self.nig_kappa0 = nig_kappa0
        self.nig_alpha0 = nig_alpha0
        self.nig_beta0 = nig_beta0
        self.decision_rule = decision_rule
        self.short_run_k = short_run_k
        self.verbose = verbose


class NIGState:
    """充分统计量 (n, S, Q)，足以还原 NIG 后验。"""

    __slots__ = ("n", "S", "Q")

    def __init__(self, n=0, S=0.0, Q=0.0):
        self.n = n
        self.S = S
        self.Q = Q

    def update(self, x):
        return NIGState(self.n + 1, self.S + x, self.Q + x * x)


class NIGPrior:
    """Normal-Inverse-Gamma 共轭先验。

        μ | σ² ~ N(μ0, σ²/κ0)
        σ²    ~ Inv-Gamma(α0, β0)

    后验仍是 NIG，故预测分布是 Student-t，闭式可算，无需 MCMC。
    """

    def __init__(self, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def posterior_params(self, state):
        """由充分统计量得到后验 (μ, κ, α, β)。"""
        n, S, Q = state.n, state.S, state.Q
        kappa = self.kappa0 + n
        if n == 0:
            return self.mu0, self.kappa0, self.alpha0, self.beta0

        mu = (self.kappa0 * self.mu0 + S) / kappa
        alpha = self.alpha0 + n / 2.0
        ss = Q - S * S / n
        if ss < 0.0:                       # 浮点误差保护
            ss = 0.0
        x_bar = S / n
        beta = self.beta0 + 0.5 * ss + \
            0.5 * self.kappa0 * n * (x_bar - self.mu0) ** 2 / kappa
        if beta <= 0.0:
            beta = 1e-12
        return mu, kappa, alpha, beta

    def predictive_logpdf(self, x, state):
        """Student-t 预测密度 log p(x | 段内统计量)。"""
        mu, kappa, alpha, beta = self.posterior_params(state)
        nu = 2.0 * alpha
        scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
        if scale_sq <= 1e-30:
            return -1e10 if abs(x - mu) > 1e-6 else 1e10
        scale = np.sqrt(scale_sq)
        z = (x - mu) / scale
        return (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
            - np.log(scale)
            - (nu + 1.0) / 2.0 * np.log1p(z * z / nu)
        )


class BOCPDDetector:
    """在线 BOCPD 检测器。

    用法::

        det = BOCPDDetector(BOCPDConfig())
        cps = det.fit(returns)           # 回溯检测
        step = det.online_step(x_new)    # 实盘：每来一个数据点调用一次
    """

    def __init__(self, config=None, log_callback=None):
        self.config = config or BOCPDConfig()
        self.log = log_callback or print
        self.hazard = float(self.config.hazard)

        self.prior = NIGPrior(
            mu0=self.config.nig_mu0,
            kappa0=self.config.nig_kappa0,
            alpha0=self.config.nig_alpha0,
            beta0=self.config.nig_beta0,
        )
        self.max_run_length = int(self.config.max_run_length)

        self._R = None
        self._states = None
        self._t = 0
        self._last_short_mass = 0.0

        self.changepoint_probs = None
        self.map_run_lengths = None
        self.short_run_probs = None
        self.log_evidence = None
        self.detected_cps = []

    # ---------- 初始化 ----------
    def _init_prior_from_data(self, data):
        """用全样本均值/方差把先验中心化，避免尺度不匹配导致的假变点。"""
        if len(data) == 0:
            return
        mu = float(np.mean(data))
        var = float(np.var(data))
        self.prior.mu0 = mu
        self.prior.beta0 = max(var * self.prior.alpha0, 1e-12)

    def _init_online(self):
        self._R = np.array([1.0])          # 运行长度后验，初始 r=0 概率 1
        self._states = [NIGState()]
        self._t = 0

    def reset(self):
        self._init_online()

    # ---------- 核心递推 ----------
    def update(self, x):
        """吃进一个观测，返回 (P(变点), MAP 运行长度, log 预测证据)。"""
        if self._R is None:
            self._init_online()

        H = self.hazard

        # 每个候选运行长度下的预测似然
        log_pred = np.array(
            [self.prior.predictive_logpdf(x, s) for s in self._states]
        )
        log_R = np.log(self._R + 1e-300)
        log_joint = log_R + log_pred

        # 归一化常数：x_t 的证据
        log_evidence = logsumexp(log_joint)

        # growth：延续当前段；reset：发生变点，运行长度归零
        log_growth = log_joint + log(1.0 - H)
        log_reset = log_evidence + log(H)

        log_new_R = np.empty(len(self._R) + 1)
        log_new_R[0] = log_reset
        log_new_R[1:] = log_growth
        log_new_R -= log_evidence         # 全部在对数域归一化

        new_R = np.exp(log_new_R)

        new_states = [NIGState()]
        for s in self._states:
            new_states.append(s.update(x))

        # 截断过长运行长度，控制内存与耗时
        if len(new_R) > self.max_run_length:
            new_R = new_R[:self.max_run_length]
            new_states = new_states[:self.max_run_length]
            s = new_R.sum()
            if s > 0:
                new_R /= s

        self._R = new_R
        self._states = new_states
        self._t += 1

        # 常量 hazard 下 new_R[0] 恒等于 H；真正携带信号的是短运行长度上的概率质量
        k = int(self.config.short_run_k)
        self._last_short_mass = float(new_R[:k].sum())

        return float(new_R[0]), int(np.argmax(new_R)), float(log_evidence)

    # ---------- 批量回溯 ----------
    def fit(self, data):
        """对整条序列重放在线递推，并做后处理得到变点位置。"""
        if data is None or len(data) == 0:
            self.changepoint_probs = np.zeros(0)
            self.map_run_lengths = np.zeros(0, dtype=int)
            self.short_run_probs = np.zeros(0)
            self.log_evidence = np.zeros(0)
            self.detected_cps = []
            return []

        self._init_prior_from_data(data)
        self._init_online()

        T = len(data)
        cp_probs = np.zeros(T)
        map_rl = np.zeros(T, dtype=int)
        short_p = np.zeros(T)
        log_evs = np.zeros(T)

        for t in range(T):
            p, r, le = self.update(float(data[t]))
            cp_probs[t] = p
            map_rl[t] = r
            short_p[t] = self._last_short_mass
            log_evs[t] = le

        self.changepoint_probs = cp_probs
        self.map_run_lengths = map_rl
        self.short_run_probs = short_p
        self.log_evidence = log_evs

        self.detected_cps = self.detect_changepoints()
        return self.detected_cps

    def detect_changepoints(self, mode=None, threshold=None, min_gap=None):
        """把后验序列转成离散变点。

        mode="run_length"：P(r_t ≤ k | x_{1:t}) 超过阈值 —— 表示「当前段很可能
        刚刚重启」，是常量 hazard 下唯一有信号量的判据。
        mode="r0"：原 notebook 的 R[0] 阈值法（此处仅作对照）。

        两者都用「峰优先 + 最小间隔去重」抑制相邻抖动。
        """
        mode = mode or getattr(self.config, "decision_rule", "run_length")
        if mode == "r0":
            probs = self.changepoint_probs
        else:
            probs = self.short_run_probs
        if probs is None or len(probs) == 0:
            return []

        threshold = self.config.cp_threshold if threshold is None else threshold
        min_gap = int(self.config.min_gap if min_gap is None else min_gap)
        k = int(self.config.short_run_k)

        # 前 k 期运行长度本身就不可能超过 t，属于启动伪影，直接跳过
        start = k if mode != "r0" else 0
        candidates = [t for t in range(start, len(probs)) if probs[t] > threshold]
        candidates.sort(key=lambda t: probs[t], reverse=True)

        selected = []
        for t in candidates:
            if all(abs(t - s) >= min_gap for s in selected):
                selected.append(t)

        return sorted(selected)

    # ---------- 在线接口 ----------
    def online_step(self, x):
        cp_prob, map_r, log_ev = self.update(float(x))
        score = (cp_prob if getattr(self.config, "decision_rule", "run_length") == "r0"
                 else self._last_short_mass)
        return {
            "t": self._t,
            "cp_prob": cp_prob,                 # R[0]，常量 hazard 下恒为 H
            "short_run_prob": self._last_short_mass,
            "map_run_length": map_r,
            "log_evidence": log_ev,
            "is_changepoint": score > self.config.cp_threshold,
        }

    def recent_warning(self, probs, window=8):
        """最近 window 期的在线预警摘要。"""
        if probs is None:
            return {}
        probs = np.asarray(probs)
        T = len(probs)
        if T == 0:
            return {}
        w = min(window, T)
        recent = probs[-w:]

        threshold = self.config.cp_threshold
        recent_peaks = [T - w + i for i in range(w) if recent[i] > threshold]
        last_cp_in_window = recent_peaks[-1] if recent_peaks else None

        return {
            "current_cp_prob": float(probs[-1]),
            "recent_cp_probs": recent.tolist(),
            "in_new_segment": bool(probs[-1] > threshold),
            "last_cp_in_recent": last_cp_in_window,
            "window": w,
        }
