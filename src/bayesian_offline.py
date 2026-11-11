# -*- coding: utf-8 -*-
"""
离线贝叶斯多变点估计 —— 乘积划分模型（Product Partition Model）精确后验

模型
----
数据 x_1..x_n 被变点 0 = τ_0 < τ_1 < ... < τ_m < τ_{m+1} = n 切成若干段。
每段独立服从一个 AR(p) 模型：

    y_t = Σ_{i=1..p} β_i y_{t-i} + ε_t,   ε_t ~ N(0, σ²)

先验：β ~ N(0, δ²I)，σ² ~ Inv-Gamma(v/2, γ/2)。
β、σ² 可解析积分掉，得到每段的**边际似然** P(s, t, p)。

段长先验取几何分布（等价于变点以 hazard λ 独立发生）：
    新段从 t+1 开始 = λ · (1-λ)^(t-s)

推断
----
由于各段边际似然可积，整体后验可写为乘积形式，于是可以做**精确**的
前向-后向递推：

    Q(s) = Σ_{t} [ λ(1-λ)^{t-s} · P(s,t,p) / p_max · Q(t+1) ]

其中 Q(s) 是「从 s 开始到序列末尾」的证据和。利用 Q 即可从后验
**精确独立采样**变点配置（无需 MCMC），多次采样得到每个位置的后验
包含概率 P(τ = t | x)。

复杂度
------
- 边际似然矩阵：O(n · L · p_max) 段，每段 O(n_obs²)（投影矩阵）
- 前向-后向：O(n · L · p_max)
- 采样：O(k · m · L · p_max)

其中 L = max_segment_length。相比 BOCPD，本方法给出的是一致（同时考虑
全部变点组合）的后验，而不是逐点在线概率。
"""

import numpy as np
from scipy.special import gammaln

__all__ = ["BayesConfig", "BayesianChangePointDetector"]


class BayesConfig:
    """离线贝叶斯变点检测的超参数（与 notebook 中 Config 字段一致）。"""

    def __init__(self, p_max=2, lambda_param=0.02, p_c=0.5, k=2000,
                 burn_in=0, delta=1.0, v=4, gamma=2.0,
                 max_segment_length=520, min_gap=8, verbose=True):
        self.p_max = p_max                    # AR 阶数搜索范围 1..p_max
        self.lambda_param = lambda_param      # 变点先验 hazard（段长几何分布）
        self.p_c = p_c                        # 后验包含概率阈值
        self.k = k                            # 后验采样次数
        self.burn_in = burn_in                # 预烧样本数
        self.delta = delta                    # β 先验标准差
        self.v = v                            # σ² 逆 Gamma 自由度
        self.gamma = gamma                    # σ² 逆 Gamma 尺度（默认由数据估计覆盖）
        self.max_segment_length = max_segment_length
        self.min_gap = min_gap                # 变点最小间隔
        self.verbose = verbose


class BayesianChangePointDetector:
    """离线、多变的贝叶斯变点检测器。

    用法::

        det = BayesianChangePointDetector(BayesConfig())
        cps = det.fit(returns)          # 返回变点下标
        det.posterior_probs             # 每个位置的后验包含概率
    """

    def __init__(self, config=None, log_callback=None):
        self.config = config or BayesConfig()
        self.log = log_callback or print

        self.p_max = self.config.p_max
        self.lambda_param = self.config.lambda_param
        self.p_c = self.config.p_c
        self.k = self.config.k
        self.burn_in = self.config.burn_in
        self.delta = self.config.delta
        self.v = self.config.v
        self.gamma = self.config.gamma
        self.max_segment_length = self.config.max_segment_length

        self.data = None
        self.n = None
        self.log_P_matrix = None
        self.log_Q_array = None
        self.change_points_samples = []
        self.posterior_probs = None

    def _vprint(self, msg):
        if self.config.verbose:
            self.log(msg)

    def estimate_parameters(self, data):
        """用经验方差把 σ² 先验中心化：γ = (v - 2) · σ̂²。"""
        n = len(data)
        residuals = data - np.mean(data)
        if n <= 1:
            self.gamma = 1.0
            return
        sigma_sq = np.var(residuals)
        self.gamma = max(sigma_sq * (self.v - 2), 1e-12)
        self._vprint(f"参数估计: v={self.v}, γ={self.gamma:.6e}, λ={self.lambda_param}")

    # ---------- 分段边际似然 ----------
    def _build_design_matrix(self, s, t, p):
        """构造段 [s,t] 的 AR(p) 设计矩阵 G 与响应 y。"""
        segment_length = t - s + 1
        if segment_length <= p:
            return None, None
        y = self.data[s + p: t + 1]
        n_obs = len(y)
        G = np.zeros((n_obs, p))
        for i in range(n_obs):
            G[i, :] = self.data[s + i: s + i + p][::-1]
        return G, y

    def _compute_log_P(self, s, t, p):
        """AR(p) 分段边际似然 log P(s,t,p)。

        积分掉 β 与 σ² 后：
            a = v/2, b = γ/2, Σ = (GᵀG + δ⁻²I)⁻¹
            yᵀPy = yᵀ(I - GΣGᵀ)y
            logP = -n/2·log(2π) + 0.5·log|Σ| - p·logδ
                   + a·logb - logΓ(a) + logΓ(a + n/2)
                   - (a + n/2)·log(b + yᵀPy/2)
        """
        segment_length = t - s + 1
        if segment_length <= p:
            return -np.inf

        G, y = self._build_design_matrix(s, t, p)
        if G is None:
            return -np.inf

        n_obs = len(y)
        Delta_inv = np.eye(p) / (self.delta ** 2)
        Sigma = np.linalg.inv(G.T @ G + Delta_inv)
        P_proj = np.eye(n_obs) - G @ Sigma @ G.T
        y_norm_p = float(y.T @ P_proj @ y)
        if y_norm_p < 0.0:                 # 浮点误差保护
            y_norm_p = 0.0

        a = self.v / 2.0
        b = self.gamma / 2.0

        return (
            -n_obs / 2.0 * np.log(2.0 * np.pi)
            + 0.5 * np.linalg.slogdet(Sigma)[1]
            - p * np.log(self.delta)
            + a * np.log(b)
            - gammaln(a)
            + gammaln(a + n_obs / 2.0)
            - (a + n_obs / 2.0) * np.log(b + 0.5 * y_norm_p)
        )

    def _compute_P_matrix(self):
        """预计算所有段 (s,t) 与阶数 p 的边际似然。"""
        n = self.n
        p_max = self.p_max
        L = self.max_segment_length

        self.log_P_matrix = np.full((n, n, p_max), -np.inf)

        report_step = max(1, n // 10)
        for s in range(n):
            t_end = min(s + L, n)
            for t in range(s, t_end):
                for p_idx in range(p_max):
                    p = p_idx + 1
                    self.log_P_matrix[s, t, p_idx] = self._compute_log_P(s, t, p)
            if s % report_step == 0:
                self._vprint(f"  P矩阵进度: {s}/{n}")

        self._vprint(f"P矩阵计算完成，形状: {self.log_P_matrix.shape}")

    # ---------- 前向-后向证据和 ----------
    def _forward_backward_pass(self):
        """Q(s)：从 s 起到末尾的后验证据和（对数域）。"""
        n = self.n
        p_max = self.p_max
        lambda_param = self.lambda_param
        L = self.max_segment_length

        log_Q = np.full(n + 1, -np.inf)
        log_Q[n] = 0.0

        for s in range(n - 1, -1, -1):
            log_terms = []
            # 情况1: 变点在 t（下一段从 t+1 起）
            t_end = min(s + L, n - 1)
            for t in range(s, t_end):
                for p_idx in range(p_max):
                    lp = self.log_P_matrix[s, t, p_idx]
                    if lp == -np.inf:
                        continue
                    log_terms.append(
                        np.log(lambda_param)
                        + (t - s) * np.log(1 - lambda_param)
                        + lp - np.log(p_max)
                        + log_Q[t + 1]
                    )
            # 情况2: 最后一段 [s, n-1]
            if s + L >= n:
                for p_idx in range(p_max):
                    lp = self.log_P_matrix[s, n - 1, p_idx]
                    if lp == -np.inf:
                        continue
                    log_terms.append(
                        (n - 1 - s) * np.log(1 - lambda_param)
                        + lp - np.log(p_max)
                    )
            if log_terms:
                m = max(log_terms)
                log_Q[s] = m + np.log(np.sum(np.exp([x - m for x in log_terms])))

        self.log_Q_array = log_Q
        return log_Q

    # ---------- 精确后验采样 ----------
    def _sample_change_points_once(self):
        """一次从后验采样：依 Q 递归地逐段抽样，等价于一次精确的独立抽样。"""
        n = self.n
        p_max = self.p_max
        lambda_param = self.lambda_param
        L = self.max_segment_length

        if self.log_Q_array is None:
            self._forward_backward_pass()

        log_Q = self.log_Q_array
        change_points = []
        s = 0

        while s < n:
            log_probs, positions = [], []

            t_end = min(s + L, n - 1)
            for t in range(s, t_end):
                log_prob_sum = -np.inf
                for p_idx in range(p_max):
                    lp = self.log_P_matrix[s, t, p_idx]
                    if lp == -np.inf:
                        continue
                    log_prob = (
                        np.log(lambda_param)
                        + (t - s) * np.log(1 - lambda_param)
                        + lp - np.log(p_max)
                        + log_Q[t + 1] - log_Q[s]
                    )
                    if log_prob_sum == -np.inf:
                        log_prob_sum = log_prob
                    else:
                        m = max(log_prob_sum, log_prob)
                        log_prob_sum = m + np.log(
                            np.exp(log_prob_sum - m) + np.exp(log_prob - m)
                        )
                if log_prob_sum > -np.inf:
                    log_probs.append(log_prob_sum)
                    positions.append(t)

            if s + L >= n:
                log_prob_last = -np.inf
                for p_idx in range(p_max):
                    lp = self.log_P_matrix[s, n - 1, p_idx]
                    if lp == -np.inf:
                        continue
                    log_prob = (
                        (n - 1 - s) * np.log(1 - lambda_param)
                        + lp - np.log(p_max) - log_Q[s]
                    )
                    if log_prob_last == -np.inf:
                        log_prob_last = log_prob
                    else:
                        m = max(log_prob_last, log_prob)
                        log_prob_last = m + np.log(
                            np.exp(log_prob_last - m) + np.exp(log_prob - m)
                        )
                if log_prob_last > -np.inf:
                    log_probs.append(log_prob_last)
                    positions.append(n - 1)

            if log_probs:
                m = max(log_probs)
                probs = np.exp([x - m for x in log_probs])
                probs = probs / probs.sum()
                t_sampled = np.random.choice(positions, p=probs)
                if t_sampled < n - 1:
                    change_points.append(t_sampled)
                    s = t_sampled + 1
                else:
                    break
            else:
                break

        return sorted(change_points)

    def _run_sampling(self):
        """重复 k 次后验采样，得到后验样本集合。"""
        self.change_points_samples = []
        self._vprint(f"开始后验采样: k={self.k}")
        report_step = max(1, self.k // 10)
        for i in range(self.k):
            cp = self._sample_change_points_once()
            if i >= self.burn_in:
                self.change_points_samples.append(cp)
            if i % report_step == 0:
                self._vprint(f"  采样进度: {i}/{self.k}")
        self._vprint(f"采样完成，有效样本数: {len(self.change_points_samples)}")

    def _compute_posterior_probs(self):
        """后验包含概率 P(τ = t | x) = 采样中出现频率。"""
        n = self.n
        probs = np.zeros(n)
        for sample in self.change_points_samples:
            for cp in sample:
                probs[cp] += 1
        if len(self.change_points_samples) > 0:
            probs /= len(self.change_points_samples)
        self.posterior_probs = probs
        return probs

    def _detect_change_points(self):
        """后验概率的局部极大 + 阈值 p_c + 最小间隔过滤。"""
        probs = self.posterior_probs
        n = len(probs)
        min_gap = getattr(self.config, "min_gap", max(4, n // 50))

        peaks = [i for i in range(1, n - 1)
                 if probs[i] > probs[i - 1] and probs[i] > probs[i + 1]]
        peaks.sort(key=lambda i: probs[i], reverse=True)

        filtered = []
        for cp in peaks:
            if probs[cp] <= self.p_c:
                break
            if all(abs(cp - x) >= min_gap for x in filtered):
                filtered.append(cp)

        self._vprint(f"检测到 {len(filtered)} 个变点")
        return sorted(filtered)

    # ---------- 主入口 ----------
    def fit(self, data):
        self.data = np.asarray(data, dtype=float)
        self.n = len(self.data)
        self._vprint(f"开始变点检测，数据长度: {self.n}")

        self.estimate_parameters(self.data)

        self._vprint("步骤 1/4: 计算 P(s,t,p) 矩阵")
        self._compute_P_matrix()

        self._vprint("步骤 2/4: 后验采样")
        self._run_sampling()

        self._vprint("步骤 3/4: 计算后验概率")
        self._compute_posterior_probs()

        self._vprint("步骤 4/4: 确定变点")
        return self._detect_change_points()
