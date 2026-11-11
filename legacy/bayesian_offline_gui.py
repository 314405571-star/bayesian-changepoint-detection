import os
import queue
import threading
import traceback
import webbrowser
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # 无 GUI 依赖，便于在子线程保存图片
import matplotlib.pyplot as plt
from scipy.special import gammaln

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

try:
    import baostock as bs
except ImportError:
    bs = None


# ==================== 0. 工具函数 ====================
def split_segments(n_data, change_points):
    """按价格索引变点（新段起点）切分区间，返回闭区间列表。"""
    segments = []
    start = 0
    for cp in sorted(change_points):
        if cp > start:
            segments.append((start, cp - 1))
        start = cp
    if start <= n_data - 1:
        segments.append((start, n_data - 1))
    return segments


# ==================== 1. 配置 ====================
class Config:
    def __init__(self, stock_code="sh.000001", stock_name=None):
        self.stock_code = stock_code
        self.stock_name = stock_name or self.get_stock_name(stock_code)

        # 周线
        self.frequency = "w"
        self.start_date = "2015-01-01"
        self.end_date = datetime.now().strftime('%Y-%m-%d')

        # 算法
        self.p_max = 2
        self.lambda_param = 0.02
        self.p_c = 0.5
        self.k = 2000
        self.burn_in = 0
        self.delta = 1.0
        self.v = 4
        self.gamma = 2
        self.max_segment_length = 520
        self.min_gap = 8

        # 可视化
        self.figure_size = (16, 12)
        self.dpi = 200
        self.save_figures = True
        self.output_dir = "./output"

        # 输出
        self.save_results = True
        self.generate_report = True
        self.verbose = True
        self.use_tqdm = True          # GUI 中会关闭

        # 夏普比率年化因子
        self.annual_factor = 52

    @staticmethod
    def get_stock_name(stock_code):
        names = {
            "sh.000001": "上证指数", "sh.000300": "沪深300", "sh.000905": "中证500",
            "sz.399001": "深证成指", "sz.399006": "创业板指",
            "sz.000001": "平安银行", "sz.000002": "万科A",
            "sh.600000": "浦发银行", "sh.600036": "招商银行", "sh.600519": "贵州茅台",
            "sz.000858": "五粮液", "sh.601318": "中国平安", "sz.000333": "美的集团",
            "sz.000651": "格力电器", "sh.600685": "中船防务",
        }
        return names.get(stock_code, stock_code)

    def print_config(self):
        print("=" * 50)
        print(f"股票: {self.stock_name} ({self.stock_code})")
        print(f"频率: {self.frequency}, 时间: {self.start_date} ~ {self.end_date}")
        print(f"p_max={self.p_max}, λ={self.lambda_param}, p_c={self.p_c}, k={self.k}")
        print(f"max_seg={self.max_segment_length}, min_gap={self.min_gap}")
        print("=" * 50)


# ==================== 2. 数据获取 ====================
class DataFetcher:
    @staticmethod
    def get_stock_data(config, log_callback=None):
        log = log_callback or print
        if bs is None:
            log("错误: 未安装 baostock，请先 pip install baostock")
            return None

        if config.verbose:
            log(f"正在获取 {config.stock_name}({config.stock_code}) 数据...")
            log(f"频率: {config.frequency}, 范围: {config.start_date} ~ {config.end_date}")

        lg = bs.login()
        if lg.error_code != '0':
            log(f"登录失败: {lg.error_msg}")
            return None

        try:
            rs = bs.query_history_k_data_plus(
                config.stock_code,
                "date,code,open,high,low,close,volume,amount,pctChg",
                start_date=config.start_date,
                end_date=config.end_date,
                frequency=config.frequency,
                adjustflag="2",
            )
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                log(f"未获取到 {config.stock_code} 的数据")
                return None

            df = pd.DataFrame(data_list, columns=rs.fields)
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            if config.verbose:
                log(f"成功获取 {len(df)} 条周线数据")
            return df
        except Exception as e:
            log(f"获取数据出错: {e}")
            return None
        finally:
            bs.logout()

    @staticmethod
    def prepare_data(df):
        data = df['close'].values
        data = pd.Series(data).ffill().bfill().values
        return data, df.index


# ==================== 3. 贝叶斯变点检测 ====================
class BayesianChangePointDetector:
    def __init__(self, config, log_callback=None):
        self.config = config
        self.log = log_callback or print

        self.p_max = config.p_max
        self.lambda_param = config.lambda_param
        self.p_c = config.p_c
        self.k = config.k
        self.burn_in = config.burn_in
        self.delta = config.delta
        self.v = config.v
        self.gamma = config.gamma
        self.max_segment_length = config.max_segment_length

        self.data = None
        self.n = None
        self.log_P_matrix = None
        self.log_Q_array = None
        self.change_points_samples = []
        self.posterior_probs = None

    def estimate_parameters(self, data):
        n = len(data)
        residuals = data - np.mean(data)
        if n <= 1:
            self.gamma = 1.0
            return
        sigma_sq = np.var(residuals)
        self.gamma = max(sigma_sq * (self.v - 2), 1e-12)
        if self.config.verbose:
            self.log(f"参数估计: v={self.v}, γ={self.gamma:.6e}, λ={self.lambda_param}")

    def _build_design_matrix(self, s, t, p):
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
        """
        AR(p) 分段边际似然 log P(s,t,p)。
        β ~ N(0, δ² I), σ² ~ IG(v/2, γ/2)，积分后：
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
        if y_norm_p < 0.0:
            y_norm_p = 0.0

        a = self.v / 2.0
        b = self.gamma / 2.0

        return (
            - n_obs / 2.0 * np.log(2.0 * np.pi)
            + 0.5 * np.linalg.slogdet(Sigma)[1]
            - p * np.log(self.delta)
            + a * np.log(b)
            - gammaln(a)
            + gammaln(a + n_obs / 2.0)
            - (a + n_obs / 2.0) * np.log(b + 0.5 * y_norm_p)
        )

    def _compute_P_matrix(self):
        n = self.n
        p_max = self.p_max
        L = self.max_segment_length

        self.log_P_matrix = np.full((n, n, p_max), -np.inf)

        report_step = max(1, n // 10)
        for s in range(n):
            # 段 [s,t] 长度 t-s+1 <= L  =>  t <= s+L-1  =>  t_end = s+L (exclusive)
            t_end = min(s + L, n)
            for t in range(s, t_end):
                for p_idx in range(p_max):
                    p = p_idx + 1
                    self.log_P_matrix[s, t, p_idx] = self._compute_log_P(s, t, p)
            if self.config.verbose and (s % report_step == 0):
                self.log(f"  P矩阵进度: {s}/{n}")

        if self.config.verbose:
            self.log(f"P矩阵计算完成，形状: {self.log_P_matrix.shape}")

    def _forward_backward_pass(self):
        n = self.n
        p_max = self.p_max
        lambda_param = self.lambda_param
        L = self.max_segment_length

        log_Q = np.full(n + 1, -np.inf)
        log_Q[n] = 0.0

        for s in range(n - 1, -1, -1):
            log_terms = []
            # 情况1: 变点在 t（下一段从 t+1 起），t <= n-2 且段长 <= L
            t_end = min(s + L, n - 1)   # exclusive
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
            # 情况2: 最后一段 [s, n-1]，要求 n-s <= L
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

    def _sample_change_points_once(self):
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
        self.change_points_samples = []
        if self.config.verbose:
            self.log(f"开始后验采样: k={self.k}")
        report_step = max(1, self.k // 10)
        for i in range(self.k):
            cp = self._sample_change_points_once()
            if i >= self.burn_in:
                self.change_points_samples.append(cp)
            if self.config.verbose and i % report_step == 0:
                self.log(f"  采样进度: {i}/{self.k}")
        if self.config.verbose:
            self.log(f"采样完成，有效样本数: {len(self.change_points_samples)}")

    def _compute_posterior_probs(self):
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

        if self.config.verbose:
            self.log(f"检测到 {len(filtered)} 个变点")
        return sorted(filtered)

    def fit(self, data):
        self.data = data
        self.n = len(data)
        if self.config.verbose:
            self.log(f"开始变点检测，数据长度: {self.n}")

        self.estimate_parameters(data)

        if self.config.verbose:
            self.log("步骤 1/4: 计算 P(s,t,p) 矩阵")
        self._compute_P_matrix()

        if self.config.verbose:
            self.log("步骤 2/4: 后验采样")
        self._run_sampling()

        if self.config.verbose:
            self.log("步骤 3/4: 计算后验概率")
        self._compute_posterior_probs()

        if self.config.verbose:
            self.log("步骤 4/4: 确定变点")
        return self._detect_change_points()


# ==================== 4. 可视化 ====================
class Visualizer:

    @staticmethod
    def plot_comprehensive_analysis(data, dates, change_points, posterior_probs, config):
        fig = plt.figure(figsize=config.figure_size)
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.2)

        name, code = config.stock_name, config.stock_code

        # 1. 价格 + 变点
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, data, 'b-', lw=1.5, alpha=0.8, label=f'{name} ({code})')
        for cp in change_points:
            if 0 <= cp < len(dates):
                ax1.axvline(x=dates[cp], color='red', ls='--', alpha=0.7, lw=1.5)
                ax1.plot(dates[cp], data[cp], 'ro', ms=8, mec='black', mew=1)
        ax1.set_title(f'{name} ({code}) 周线价格与变点', fontsize=14, fontweight='bold')
        ax1.set_ylabel('收盘价')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. 对数收益率
        returns = np.log(data[1:] / data[:-1])
        ret_dates = dates[1:]
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(ret_dates, returns, 'g-', lw=1.5, alpha=0.7, label='对数收益率')
        for cp in change_points:
            if 0 <= cp < len(dates):
                ax2.axvline(x=dates[cp], color='red', ls='--', alpha=0.7, lw=1.5)
        ax2.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax2.set_title(f'{name} 周对数收益率', fontsize=14, fontweight='bold')
        ax2.set_ylabel('对数收益率')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 3. 后验概率
        ax3 = fig.add_subplot(gs[2, 0])
        n_ret = len(posterior_probs)
        ax3.plot(ret_dates[:n_ret], posterior_probs, 'purple', lw=2, alpha=0.8, label='后验概率')
        for cp in change_points:
            ridx = cp - 1
            if 0 <= ridx < n_ret:
                ax3.plot(dates[cp], posterior_probs[ridx], 'ro', ms=8, mec='black', mew=1)
        ax3.axhline(config.p_c, color='orange', ls='--', lw=1.5, alpha=0.7,
                    label=f'阈值 (p_c={config.p_c})')
        ax3.set_title('变点后验概率', fontsize=12, fontweight='bold')
        ax3.set_ylabel('后验概率')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # 4. 变点间隔
        ax4 = fig.add_subplot(gs[2, 1])
        if len(change_points) >= 2:
            intervals = np.diff(sorted(change_points))
            ax4.hist(intervals, bins=min(15, max(1, len(intervals))),
                     color='lightcoral', edgecolor='black', alpha=0.7)
            ax4.axvline(np.mean(intervals), color='red', ls='--', lw=2,
                        label=f'平均间隔: {np.mean(intervals):.1f} 周')
            ax4.set_title('变点间隔分布', fontsize=12, fontweight='bold')
            ax4.set_xlabel('间隔(周)')
            ax4.set_ylabel('频数')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '变点不足', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=12)
            ax4.set_title('变点间隔分布', fontsize=12, fontweight='bold')

        # 5. 分段收益率
        segments = split_segments(len(data), change_points)
        ax5 = fig.add_subplot(gs[3, 0])
        ret_list, labels = [], []
        for i, (s, e) in enumerate(segments, 1):
            if e > s:
                ret_list.append((data[e] - data[s]) / data[s] * 100)
                labels.append(f'段{i}')
        if ret_list:
            colors = ['green' if r >= 0 else 'red' for r in ret_list]
            bars = ax5.bar(range(len(ret_list)), ret_list, color=colors,
                           edgecolor='black', alpha=0.7)
            for b, r in zip(bars, ret_list):
                h = b.get_height()
                ax5.text(b.get_x() + b.get_width() / 2., h + (0.1 if h >= 0 else -0.5),
                         f'{r:.1f}%', ha='center',
                         va='bottom' if h >= 0 else 'top', fontsize=9)
            ax5.axhline(0, color='k', lw=0.5)
            ax5.set_xticks(range(len(labels)))
            ax5.set_xticklabels(labels, rotation=45)
        ax5.set_title('分段收益率', fontsize=12, fontweight='bold')
        ax5.set_ylabel('收益率(%)')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. 市场状态散点
        ax6 = fig.add_subplot(gs[3, 1])
        rets, vols, sizes = [], [], []
        for (s, e) in segments:
            if e > s:
                seg = data[s:e + 1]
                rets.append((seg[-1] - seg[0]) / seg[0] * 100)
                vols.append(np.std(np.diff(seg) / seg[:-1]) * 100)
                sizes.append(max(20, e - s + 1))
        if rets:
            ax6.scatter(rets, vols, s=sizes, alpha=0.6, edgecolors='black')
            ax6.axvline(0, color='gray', ls='--', alpha=0.5)
            ax6.axhline(np.median(vols), color='gray', ls='--', alpha=0.5)
        ax6.set_title('市场状态分析', fontsize=12, fontweight='bold')
        ax6.set_xlabel('收益率(%)')
        ax6.set_ylabel('波动率(%)')
        ax6.grid(True, alpha=0.3)

        time_range = f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}"
        plt.suptitle(f'{name} ({code}) 周线变点检测\n{time_range}',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_simple_chart(data, dates, change_points, config):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
        name, code = config.stock_name, config.stock_code

        ax1.plot(dates, data, 'b-', lw=1.5, alpha=0.8, label=f'{name} ({code})')
        colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow', 'lavender']
        if change_points:
            segments = split_segments(len(data), change_points)
            for i, (s, e) in enumerate(segments):
                ax1.axvspan(dates[s], dates[e], alpha=0.2, color=colors[i % len(colors)])
            for cp in change_points:
                if 0 <= cp < len(dates):
                    ax1.axvline(dates[cp], color='red', lw=2, ls='--', alpha=0.7)
                    ax1.plot(dates[cp], data[cp], 'ro', ms=10, mec='black', mew=2)
        ax1.set_title(f'{name} ({code}) 周线 - 价格与变点', fontsize=14, fontweight='bold')
        ax1.set_ylabel('收盘价')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        ax2.set_title('变点信息', fontsize=14, fontweight='bold')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('变点后验概率')
        ax2.grid(True, alpha=0.3)
        for cp in change_points:
            if 0 <= cp < len(dates):
                ax2.axvline(dates[cp], color='red', lw=1, ls='--', alpha=0.5)

        info = f'变点数: {len(change_points)}\n'
        for i, cp in enumerate(sorted(change_points)[:3], 1):
            if 0 <= cp < len(dates):
                info += f'变点{i}: {dates[cp].strftime("%Y-%m-%d")}, {data[cp]:.2f}\n'
        if len(change_points) > 3:
            info += f'...还有 {len(change_points) - 3} 个'
        ax2.text(0.02, 0.95, info, transform=ax2.transAxes, fontsize=10, va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='gray', alpha=0.8))

        stats = (f'数据统计:\n长度: {len(data)} 周\n'
                 f'价格: {np.min(data):.2f} - {np.max(data):.2f}\n'
                 f'总收益: {((data[-1] - data[0]) / data[0] * 100):.2f}%')
        ax2.text(0.98, 0.95, stats, transform=ax2.transAxes, fontsize=9, va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                           edgecolor='gray', alpha=0.8))
        plt.tight_layout()
        return fig


# ==================== 5. 报告 ====================
class ReportGenerator:
    @staticmethod
    def generate_report(data, dates, change_points, config):
        name, code = config.stock_name, config.stock_code
        out = []
        out.append("=" * 60)
        out.append(f"{name} ({code}) 周线变点检测报告")
        out.append("=" * 60)
        out.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        out.append(f"数据范围: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
        out.append(f"数据长度: {len(data)} 周")

        out.append("\n【数据统计】")
        out.append(f"  价格范围: {np.min(data):.2f} ~ {np.max(data):.2f}")
        out.append(f"  最终价格: {data[-1]:.2f}")
        out.append(f"  总收益率: {((data[-1] - data[0]) / data[0] * 100):.2f}%")
        returns = np.log(data[1:] / data[:-1])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(config.annual_factor) \
            if np.std(returns) > 0 else 0.0
        out.append(f"  收益率均值: {np.mean(returns) * 100:.4f}%")
        out.append(f"  收益率标准差: {np.std(returns) * 100:.4f}%")
        out.append(f"  夏普比率(年化, √{config.annual_factor}): {sharpe:.4f}")

        out.append("\n【变点检测结果】")
        out.append(f"  变点数量: {len(change_points)}")
        for i, cp in enumerate(sorted(change_points), 1):
            if 0 <= cp < len(dates):
                out.append(f"    变点{i:2d}: {dates[cp].strftime('%Y-%m-%d')}, 价格={data[cp]:.2f}")

        if len(change_points) >= 2:
            intervals = np.diff(sorted(change_points))
            out.append("\n【变点间隔】")
            out.append(f"  平均: {np.mean(intervals):.1f} 周")
            out.append(f"  最短: {np.min(intervals):.1f} 周")
            out.append(f"  最长: {np.max(intervals):.1f} 周")
            out.append(f"  标准差: {np.std(intervals):.1f} 周")

        if change_points:
            out.append("\n【分段分析】")
            for i, (s, e) in enumerate(split_segments(len(data), change_points), 1):
                if e > s:
                    seg = data[s:e + 1]
                    ret = (seg[-1] - seg[0]) / seg[0] * 100
                    vol = np.std(np.diff(seg) / seg[:-1]) * 100
                    trend = ("📈 强劲上涨" if ret > 10 else
                             "↗️ 温和上涨" if ret > 0 else
                             "↘️ 温和下跌" if ret > -10 else "📉 大幅下跌")
                    out.append(f"  阶段{i}: {dates[s].strftime('%Y-%m-%d')} ~ "
                               f"{dates[e].strftime('%Y-%m-%d')} | "
                               f"{e - s + 1}周 | 收益 {ret:+.2f}% | "
                               f"波动 {vol:.2f}% | {trend}")

        out.append("\n【参数】")
        out.append(f"  p_max={config.p_max}, λ={config.lambda_param}, p_c={config.p_c}, k={config.k}")
        out.append(f"  max_segment_length={config.max_segment_length}, min_gap={config.min_gap}")
        out.append("=" * 60)
        return "\n".join(out)


# ==================== 6. 分析主流程（无 UI 依赖） ====================
class ChangePointAnalysis:
    def __init__(self, config=None, log_callback=None):
        self.config = config or Config()
        self.log_callback = log_callback or print
        self.data = None
        self.dates = None
        self.returns = None
        self.change_points = None
        self.posterior_probs = None
        self.figure_paths = []

    def _log(self, msg):
        self.log_callback(msg)

    def run_analysis(self):
        self._log("=" * 60)
        self._log(f"开始分析: {self.config.stock_name} ({self.config.stock_code})")
        self._log(f"频率={self.config.frequency}, {self.config.start_date} ~ {self.config.end_date}")
        self._log("=" * 60)

        os.makedirs(self.config.output_dir, exist_ok=True)

        df = DataFetcher.get_stock_data(self.config, log_callback=self._log)
        if df is None or len(df) == 0:
            self._log(f"未获取到 {self.config.stock_code} 数据")
            return None

        self.data, self.dates = DataFetcher.prepare_data(df)
        if len(self.data) < self.config.p_max + 10:
            self._log("数据太短，无法可靠检测")
            return None

        self._log(f"数据长度: {len(self.data)} 周, "
                  f"时间: {self.dates[0].date()} ~ {self.dates[-1].date()}")

        self.returns = np.log(self.data[1:] / self.data[:-1])

        detector = BayesianChangePointDetector(self.config, log_callback=self._log)
        detected = detector.fit(self.returns)
        self.posterior_probs = detector.posterior_probs

        # 收益率索引 -> 价格索引（新段起点）
        self.change_points = [cp + 1 for cp in detected if cp + 1 < len(self.data)]

        self._log(f"收益率序列变点数: {len(detected)}, 价格序列变点数: {len(self.change_points)}")

        if self.config.save_figures:
            self._log("生成图表...")
            try:
                fig1 = Visualizer.plot_comprehensive_analysis(
                    self.data, self.dates, self.change_points,
                    self.posterior_probs, self.config)
                p1 = os.path.join(self.config.output_dir,
                                  f"{self.config.stock_code}_w_analysis.png")
                fig1.savefig(p1, dpi=self.config.dpi, bbox_inches='tight')
                plt.close(fig1)

                fig2 = Visualizer.plot_simple_chart(
                    self.data, self.dates, self.change_points, self.config)
                p2 = os.path.join(self.config.output_dir,
                                  f"{self.config.stock_code}_w_simple.png")
                fig2.savefig(p2, dpi=self.config.dpi, bbox_inches='tight')
                plt.close(fig2)

                self.figure_paths = [p1, p2]
                self._log(f"图表: {p1}")
                self._log(f"图表: {p2}")
            except Exception as e:
                self._log(f"生成图表失败: {e}")

        if self.config.save_results:
            try:
                results_df = pd.DataFrame({
                    'date': self.dates,
                    'close': self.data,
                    'is_changepoint': False,
                })
                for cp in self.change_points:
                    if 0 <= cp < len(results_df):
                        results_df.loc[cp, 'is_changepoint'] = True
                results_path = os.path.join(
                    self.config.output_dir, f"{self.config.stock_code}_w_results.csv")
                results_df.to_csv(results_path, index=False)

                if self.posterior_probs is not None:
                    prob_df = pd.DataFrame({
                        'date': self.dates[1:1 + len(self.posterior_probs)],
                        'posterior_prob': self.posterior_probs,
                    })
                    prob_path = os.path.join(
                        self.config.output_dir,
                        f"{self.config.stock_code}_w_posterior_probs.csv")
                    prob_df.to_csv(prob_path, index=False)

                self._log(f"结果 CSV: {results_path}")
            except Exception as e:
                self._log(f"保存结果失败: {e}")

        if self.config.generate_report:
            try:
                report = ReportGenerator.generate_report(
                    self.data, self.dates, self.change_points, self.config)
                rp = os.path.join(self.config.output_dir,
                                  f"{self.config.stock_code}_w_report.txt")
                with open(rp, 'w', encoding='utf-8') as f:
                    f.write(report)
                self._log(f"报告: {rp}")
            except Exception as e:
                self._log(f"保存报告失败: {e}")

        self._log("分析完成。")
        return self

    def build_report(self):
        if self.data is None:
            return "无结果"
        return ReportGenerator.generate_report(
            self.data, self.dates, self.change_points, self.config)


# ==================== 7. GUI ====================
COMMON_STOCKS = [
    "sh.000001", "sh.000300", "sh.000905",
    "sz.399001", "sz.399006",
    "sh.600519", "sh.600036", "sh.600685",
    "sz.000001", "sz.000858", "sh.601318",
    "sz.000333", "sz.000651",
]


class ChangePointGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("贝叶斯变点检测系统 - 周线")
        self.root.geometry("960x760")

        self.log_queue = queue.Queue()
        self.worker = None

        self._build_ui()
        self._poll_queue()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.LabelFrame(self.root, text="参数设置", padding=8)
        top.pack(fill='x', padx=8, pady=6)

        # 行 1：股票代码
        r1 = ttk.Frame(top)
        r1.pack(fill='x', pady=2)
        ttk.Label(r1, text="股票代码:").pack(side='left')
        self.code_var = tk.StringVar(value="sh.600685")
        self.code_combo = ttk.Combobox(r1, textvariable=self.code_var,
                                       values=COMMON_STOCKS, width=18)
        self.code_combo.pack(side='left', padx=(4, 16))
        self.code_combo.bind("<<ComboboxSelected>>", self._on_code_selected)

        ttk.Label(r1, text="股票名称:").pack(side='left')
        self.name_var = tk.StringVar(value=Config.get_stock_name("sh.600685"))
        ttk.Entry(r1, textvariable=self.name_var, width=16,
                  state='readonly').pack(side='left', padx=4)

        # 行 2：日期
        r2 = ttk.Frame(top)
        r2.pack(fill='x', pady=2)
        ttk.Label(r2, text="开始日期:").pack(side='left')
        self.start_var = tk.StringVar(value="2015-01-01")
        ttk.Entry(r2, textvariable=self.start_var, width=14).pack(side='left', padx=4)
        ttk.Label(r2, text="结束日期:").pack(side='left', padx=(16, 0))
        self.end_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        ttk.Entry(r2, textvariable=self.end_var, width=14).pack(side='left', padx=4)

        # 高级参数区
        self.advanced_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="显示高级参数",
                        variable=self.advanced_visible,
                        command=self._toggle_advanced).pack(anchor='w', pady=(6, 0))

        self.advanced_frame = ttk.Frame(top)
        # 不 pack，等切换时再 pack

        adv = self.advanced_frame
        row = ttk.Frame(adv)
        row.pack(fill='x', pady=2)
        ttk.Label(row, text="p_max:").pack(side='left')
        self.pmax_var = tk.StringVar(value="2")
        ttk.Entry(row, textvariable=self.pmax_var, width=6).pack(side='left', padx=(2, 14))
        ttk.Label(row, text="lambda:").pack(side='left')
        self.lambda_var = tk.StringVar(value="0.02")
        ttk.Entry(row, textvariable=self.lambda_var, width=8).pack(side='left', padx=(2, 14))
        ttk.Label(row, text="p_c:").pack(side='left')
        self.pc_var = tk.StringVar(value="0.5")
        ttk.Entry(row, textvariable=self.pc_var, width=6).pack(side='left', padx=(2, 14))
        ttk.Label(row, text="k:").pack(side='left')
        self.k_var = tk.StringVar(value="2000")
        ttk.Entry(row, textvariable=self.k_var, width=8).pack(side='left', padx=(2, 14))

        row2 = ttk.Frame(adv)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="max_segment_length:").pack(side='left')
        self.maxseg_var = tk.StringVar(value="520")
        ttk.Entry(row2, textvariable=self.maxseg_var, width=8).pack(side='left', padx=(2, 14))
        ttk.Label(row2, text="min_gap:").pack(side='left')
        self.mingap_var = tk.StringVar(value="8")
        ttk.Entry(row2, textvariable=self.mingap_var, width=6).pack(side='left', padx=(2, 14))

        # 按钮栏
        bar = ttk.Frame(self.root)
        bar.pack(fill='x', padx=8, pady=4)
        self.run_btn = ttk.Button(bar, text="开始分析", command=self._start_analysis)
        self.run_btn.pack(side='left')
        ttk.Button(bar, text="清空日志", command=self._clear_log).pack(side='left', padx=6)
        ttk.Button(bar, text="打开输出目录", command=self._open_output).pack(side='left')

        self.progress = ttk.Progressbar(bar, mode='indeterminate', length=200)
        self.progress.pack(side='right')

        # Notebook: 日志 / 报告
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=8, pady=6)

        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="日志")
        self.log_text = scrolledtext.ScrolledText(log_tab, wrap='word',
                                                  font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True)
        self.log_text.configure(state='disabled')

        report_tab = ttk.Frame(self.notebook)
        self.notebook.add(report_tab, text="报告")
        self.report_text = scrolledtext.ScrolledText(report_tab, wrap='word',
                                                     font=('Consolas', 9))
        self.report_text.pack(fill='both', expand=True)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           relief='sunken', anchor='w')
        status.pack(fill='x', padx=8, pady=(0, 6))

    def _toggle_advanced(self):
        if self.advanced_visible.get():
            self.advanced_frame.pack(fill='x', pady=4)
        else:
            self.advanced_frame.pack_forget()

    def _on_code_selected(self, _event=None):
        code = self.code_var.get().strip()
        self.name_var.set(Config.get_stock_name(code))

    # ---------- 交互逻辑 ----------
    def _append_log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', msg + "\n")
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

    def _open_output(self):
        path = os.path.abspath("./output")
        os.makedirs(path, exist_ok=True)
        webbrowser.open("file://" + path)

    def _start_analysis(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("提示", "分析正在进行中，请稍候...")
            return

        code = self.code_var.get().strip()
        if not code:
            messagebox.showwarning("提示", "请输入股票代码，如 sh.600685")
            return

        start = self.start_var.get().strip()
        end = self.end_var.get().strip()
        try:
            datetime.strptime(start, '%Y-%m-%d')
            datetime.strptime(end, '%Y-%m-%d')
        except ValueError:
            messagebox.showwarning("提示", "日期格式应为 YYYY-MM-DD")
            return

        config = Config(stock_code=code)
        config.start_date = start
        config.end_date = end

        if self.advanced_visible.get():
            try:
                config.p_max = int(self.pmax_var.get())
                config.lambda_param = float(self.lambda_var.get())
                config.p_c = float(self.pc_var.get())
                config.k = int(self.k_var.get())
                config.max_segment_length = int(self.maxseg_var.get())
                config.min_gap = int(self.mingap_var.get())
            except ValueError:
                messagebox.showwarning("提示", "高级参数格式错误")
                return

        config.use_tqdm = False
        config.verbose = True

        self.run_btn.config(state='disabled')
        self.progress.start(10)
        self.status_var.set("分析中...")
        self._append_log("\n" + "#" * 60)
        self._append_log(f"启动分析: {config.stock_name} ({config.stock_code})")

        self.worker = threading.Thread(
            target=self._run_worker, args=(config,), daemon=True)
        self.worker.start()

    def _run_worker(self, config):
        try:
            analyzer = ChangePointAnalysis(config, log_callback=self._thread_log)
            result = analyzer.run_analysis()
            self.log_queue.put(('done', result))
        except Exception:
            self.log_queue.put(('error', traceback.format_exc()))

    def _thread_log(self, msg):
        self.log_queue.put(('log', msg))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == 'log':
                    self._append_log(payload)
                elif kind == 'done':
                    self._on_done(payload)
                elif kind == 'error':
                    self._on_error(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _finish_ui(self):
        self.progress.stop()
        self.run_btn.config(state='normal')

    def _on_done(self, analyzer):
        self._finish_ui()
        if analyzer is None:
            self.status_var.set("分析失败")
            messagebox.showerror("失败", "未能获取数据或数据不足")
            return

        self.status_var.set("分析完成")
        try:
            report = analyzer.build_report()
            self.report_text.delete('1.0', 'end')
            self.report_text.insert('end', report)
            self.notebook.select(1)      # 切到报告页
        except Exception as e:
            self._append_log(f"报告生成失败: {e}")

        if analyzer.figure_paths:
            try:
                webbrowser.open("file://" + os.path.abspath(analyzer.figure_paths[0]))
            except Exception:
                pass

    def _on_error(self, tb):
        self._finish_ui()
        self.status_var.set("发生错误")
        self._append_log("错误:\n" + tb)
        messagebox.showerror("错误", "分析出现异常，请查看日志")


# ==================== 8. 入口 ====================
def main():
    root = tk.Tk()
    ChangePointGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()