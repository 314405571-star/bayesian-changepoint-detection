import os
import queue
import threading
import traceback
import webbrowser
import warnings
from datetime import datetime
from math import log, pi

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gammaln, logsumexp

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
    segments = []
    start = 0
    for cp in sorted(change_points or []):
        if cp > start:
            segments.append((start, cp - 1))
        start = cp
    if start <= n_data - 1:
        segments.append((start, n_data - 1))
    return segments


def safe_last(arr, default=None):
    """安全访问数组最后一个元素。"""
    if arr is None:
        return default
    try:
        if len(arr) == 0:
            return default
    except TypeError:
        return default
    return arr[-1]


# ==================== 1. 配置 ====================
class Config:
    def __init__(self, stock_code="sh.000001", stock_name=None):
        self.stock_code = stock_code
        self.stock_name = stock_name or self.get_stock_name(stock_code)

        self.frequency = "w"
        self.start_date = "2015-01-01"
        self.end_date = datetime.now().strftime('%Y-%m-%d')

        self.hazard = 1.0 / 50.0
        self.cp_threshold = 0.5
        self.min_gap = 8
        self.max_run_length = 5000

        self.nig_mu0 = 0.0
        self.nig_kappa0 = 1.0
        self.nig_alpha0 = 1.0
        self.nig_beta0 = 1e-4

        self.figure_size = (16, 14)
        self.dpi = 200
        self.save_figures = True
        self.output_dir = "./output"

        self.save_results = True
        self.generate_report = True
        self.verbose = True
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
        print("=" * 60)
        print(f"股票: {self.stock_name} ({self.stock_code})")
        print(f"频率: {self.frequency}, 时间: {self.start_date} ~ {self.end_date}")
        print(f"H={self.hazard:.4f} (λ={1/self.hazard:.1f}), "
              f"阈值={self.cp_threshold}, min_gap={self.min_gap}")
        print("=" * 60)


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


# ==================== 3. BOCPD 算法 ====================
class NIGState:
    __slots__ = ('n', 'S', 'Q')

    def __init__(self, n=0, S=0.0, Q=0.0):
        self.n = n
        self.S = S
        self.Q = Q

    def update(self, x):
        return NIGState(self.n + 1, self.S + x, self.Q + x * x)


class NIGPrior:
    def __init__(self, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def posterior_params(self, state):
        n, S, Q = state.n, state.S, state.Q
        kappa = self.kappa0 + n
        if n == 0:
            return self.mu0, self.kappa0, self.alpha0, self.beta0

        mu = (self.kappa0 * self.mu0 + S) / kappa
        alpha = self.alpha0 + n / 2.0
        ss = Q - S * S / n
        if ss < 0.0:
            ss = 0.0
        x_bar = S / n
        beta = self.beta0 + 0.5 * ss + \
            0.5 * self.kappa0 * n * (x_bar - self.mu0) ** 2 / kappa
        if beta <= 0.0:
            beta = 1e-12
        return mu, kappa, alpha, beta

    def predictive_logpdf(self, x, state):
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
    def __init__(self, config, log_callback=None):
        self.config = config
        self.log = log_callback or print
        self.hazard = float(config.hazard)

        self.prior = NIGPrior(
            mu0=config.nig_mu0,
            kappa0=config.nig_kappa0,
            alpha0=config.nig_alpha0,
            beta0=config.nig_beta0,
        )
        self.max_run_length = int(config.max_run_length)

        self._R = None
        self._states = None
        self._t = 0

        self.changepoint_probs = None
        self.map_run_lengths = None
        self.log_evidence = None
        self.detected_cps = []

    def _init_prior_from_data(self, data):
        if len(data) == 0:
            return
        mu = float(np.mean(data))
        var = float(np.var(data))
        self.prior.mu0 = mu
        self.prior.beta0 = max(var * self.prior.alpha0, 1e-12)

    def _init_online(self):
        self._R = np.array([1.0])
        self._states = [NIGState()]
        self._t = 0

    def reset(self):
        self._init_online()

    def update(self, x):
        if self._R is None:
            self._init_online()

        H = self.hazard
        log_pred = np.array(
            [self.prior.predictive_logpdf(x, s) for s in self._states]
        )
        log_R = np.log(self._R + 1e-300)
        log_joint = log_R + log_pred

        log_evidence = logsumexp(log_joint)

        log_growth = log_joint + log(1.0 - H)
        log_reset = log_evidence + log(H)

        log_new_R = np.empty(len(self._R) + 1)
        log_new_R[0] = log_reset
        log_new_R[1:] = log_growth
        log_new_R -= log_evidence

        new_R = np.exp(log_new_R)

        new_states = [NIGState()]
        for s in self._states:
            new_states.append(s.update(x))

        if len(new_R) > self.max_run_length:
            new_R = new_R[:self.max_run_length]
            new_states = new_states[:self.max_run_length]
            s = new_R.sum()
            if s > 0:
                new_R /= s

        self._R = new_R
        self._states = new_states
        self._t += 1

        return float(new_R[0]), int(np.argmax(new_R)), float(log_evidence)

    def fit(self, data, log_evidence_history=True):
        if data is None or len(data) == 0:
            self.changepoint_probs = np.zeros(0)
            self.map_run_lengths = np.zeros(0, dtype=int)
            self.log_evidence = np.zeros(0)
            self.detected_cps = []
            return []

        self._init_prior_from_data(data)
        self._init_online()

        T = len(data)
        cp_probs = np.zeros(T)
        map_rl = np.zeros(T, dtype=int)
        log_evs = np.zeros(T)

        for t in range(T):
            p, r, le = self.update(float(data[t]))
            cp_probs[t] = p
            map_rl[t] = r
            log_evs[t] = le

        self.changepoint_probs = cp_probs
        self.map_run_lengths = map_rl
        self.log_evidence = log_evs if log_evidence_history else None

        self.detected_cps = self._detect_changepoints(cp_probs)
        return self.detected_cps

    def _detect_changepoints(self, probs):
        if probs is None or len(probs) == 0:
            return []
        threshold = self.config.cp_threshold
        min_gap = int(self.config.min_gap)

        candidates = [t for t in range(len(probs)) if probs[t] > threshold]
        candidates.sort(key=lambda t: probs[t], reverse=True)

        selected = []
        for t in candidates:
            if all(abs(t - s) >= min_gap for s in selected):
                selected.append(t)

        return sorted(selected)

    def online_step(self, x):
        cp_prob, map_r, log_ev = self.update(float(x))
        return {
            't': self._t,
            'cp_prob': cp_prob,
            'map_run_length': map_r,
            'log_evidence': log_ev,
            'is_changepoint': cp_prob > self.config.cp_threshold,
        }

    def recent_warning(self, probs, window=8):
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
            'current_cp_prob': float(probs[-1]),
            'recent_cp_probs': recent.tolist(),
            'in_new_segment': bool(probs[-1] > threshold),
            'last_cp_in_recent': last_cp_in_window,
            'window': w,
        }


# ==================== 4. 可视化 ====================
class Visualizer:

    @staticmethod
    def plot_comprehensive_analysis(data, dates, change_points,
                                    posterior_probs, map_rl, config):
        # 【修复】change_points 为空安全；不再叠加"未检测到变点"提示框
        change_points = list(change_points or [])
        fig = plt.figure(figsize=config.figure_size)
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.2)

        name, code = config.stock_name, config.stock_code
        ret_dates = dates[1:]
        n_ret = len(ret_dates)

        # 1) 价格
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, data, 'b-', lw=1.5, alpha=0.8, label=f'{name} ({code})')
        for cp in change_points:
            if 0 <= cp < len(dates):
                ax1.axvline(dates[cp], color='red', ls='--', alpha=0.7, lw=1.5)
                ax1.plot(dates[cp], data[cp], 'ro', ms=8, mec='black', mew=1)
        ax1.set_title(f'{name} ({code}) 周线价格与变点', fontsize=14, fontweight='bold')
        ax1.set_ylabel('收盘价')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2) 收益率
        returns = np.log(data[1:] / data[:-1])
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(ret_dates, returns, 'g-', lw=1.5, alpha=0.7, label='对数收益率')
        for cp in change_points:
            if 0 <= cp < len(dates):
                ax2.axvline(dates[cp], color='red', ls='--', alpha=0.7, lw=1.5)
        ax2.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax2.set_title(f'{name} 周对数收益率', fontsize=14, fontweight='bold')
        ax2.set_ylabel('对数收益率')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 3) 变点概率 R[0]
        ax3 = fig.add_subplot(gs[2, 0])
        if posterior_probs is not None and len(posterior_probs) > 0:
            pp = np.asarray(posterior_probs)
            if len(pp) > n_ret:
                pp = pp[:n_ret]
                plot_dates = ret_dates
            else:
                plot_dates = ret_dates[:len(pp)]
            ax3.plot(plot_dates, pp, 'purple', lw=1.8, alpha=0.9, label='R[0] (变点概率)')
            for cp in change_points:
                ridx = cp - 1
                if 0 <= ridx < len(pp):
                    ax3.plot(dates[cp], pp[ridx], 'ro', ms=7, mec='black', mew=1)
            ax3.axhline(config.cp_threshold, color='orange', ls='--', lw=1.5,
                        alpha=0.8, label=f'阈值 ({config.cp_threshold})')
        ax3.set_title('在线变点概率 R[0]', fontsize=12, fontweight='bold')
        ax3.set_ylabel('R[0]')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # 4) MAP 运行长度
        ax4 = fig.add_subplot(gs[2, 1])
        if map_rl is not None and len(map_rl) > 0:
            rl = np.asarray(map_rl)
            if len(rl) > n_ret:
                rl = rl[:n_ret]
                plot_dates = ret_dates
            else:
                plot_dates = ret_dates[:len(rl)]
            ax4.plot(plot_dates, rl, 'teal', lw=1.5, alpha=0.9)
        ax4.set_title('MAP 运行长度', fontsize=12, fontweight='bold')
        ax4.set_ylabel('MAP run length (周)')
        ax4.grid(True, alpha=0.3)

        # 5) 分段收益率
        ax5 = fig.add_subplot(gs[3, 0])
        segments = split_segments(len(data), change_points)
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
                ax5.text(b.get_x() + b.get_width() / 2.,
                         h + (0.1 if h >= 0 else -0.5),
                         f'{r:.1f}%', ha='center',
                         va='bottom' if h >= 0 else 'top', fontsize=9)
            ax5.axhline(0, color='k', lw=0.5)
            ax5.set_xticks(range(len(labels)))
            ax5.set_xticklabels(labels, rotation=45)
        ax5.set_title('分段收益率', fontsize=12, fontweight='bold')
        ax5.set_ylabel('收益率(%)')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6) 市场状态散点
        ax6 = fig.add_subplot(gs[3, 1])
        rets, vols, sizes = [], [], []
        for (s, e) in segments:
            if e > s:
                seg = data[s:e + 1]
                rets.append((seg[-1] - seg[0]) / seg[0] * 100)
                with np.errstate(divide='ignore', invalid='ignore'):
                    diff = np.diff(seg) / seg[:-1]
                vols.append(np.nanstd(diff) * 100 if len(diff) > 0 else 0.0)
                sizes.append(max(20, e - s + 1))
        if rets:
            ax6.scatter(rets, vols, s=sizes, alpha=0.6, edgecolors='black')
            ax6.axvline(0, color='gray', ls='--', alpha=0.5)
            if vols:
                ax6.axhline(np.nanmedian(vols), color='gray', ls='--', alpha=0.5)
        ax6.set_title('市场状态分析', fontsize=12, fontweight='bold')
        ax6.set_xlabel('收益率(%)')
        ax6.set_ylabel('波动率(%)')
        ax6.grid(True, alpha=0.3)

        time_range = f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}"
        plt.suptitle(f'{name} ({code}) BOCPD 在线变点检测\n{time_range}',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_simple_chart(data, dates, change_points, config):
        # 【修复】change_points 为空安全；恢复原 info 文本格式
        change_points = list(change_points or [])
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
        ax2.grid(True, alpha=0.3)
        for cp in change_points:
            if 0 <= cp < len(dates):
                ax2.axvline(dates[cp], color='red', lw=1, ls='--', alpha=0.5)

        # 【恢复原格式】不再加"未检测到变点..."的 else 分支
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
        ax2.text(0.98, 0.95, stats, transform=ax2.transAxes, fontsize=9,
                 va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                           edgecolor='gray', alpha=0.8))
        plt.tight_layout()
        return fig


# ==================== 5. 报告 ====================
class ReportGenerator:

    @staticmethod
    def generate_report(data, dates, change_points, detector, config):
        """
        【修复 + 恢复原格式】
        - 保留防御式修复：空数组安全访问、索引边界检查；
        - 恢复原版输出：0 变点时报告里只有 "变点数量: 0"，
          分段分析 / 在线预警 等段落沿用原版"有数据才显示"逻辑。
        """
        change_points = list(change_points or [])
        name, code = config.stock_name, config.stock_code

        out = []
        out.append("=" * 60)
        out.append(f"{name} ({code}) BOCPD 在线变点检测报告")
        out.append("=" * 60)
        out.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        out.append(f"数据范围: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
        out.append(f"数据长度: {len(data)} 周")

        # 数据统计
        out.append("\n【数据统计】")
        out.append(f"  价格范围: {np.min(data):.2f} ~ {np.max(data):.2f}")
        out.append(f"  最终价格: {data[-1]:.2f}")
        out.append(f"  总收益率: {((data[-1] - data[0]) / data[0] * 100):.2f}%")
        if len(data) >= 2:
            returns = np.log(data[1:] / data[:-1])
            std = np.std(returns)
            sharpe = (np.mean(returns) / std * np.sqrt(config.annual_factor)
                      if std > 0 else 0.0)
            out.append(f"  收益率均值: {np.mean(returns) * 100:.4f}%")
            out.append(f"  收益率标准差: {std * 100:.4f}%")
            out.append(f"  夏普比率(年化,√{config.annual_factor}): {sharpe:.4f}")
        else:
            out.append("  数据点过少，无法计算收益率统计")

        # 变点检测结果（恢复原格式：0 变点只有一行"变点数量: 0"）
        out.append("\n【BOCPD 变点检测结果】")
        out.append(f"  变点数量: {len(change_points)}")

        cps_probs = getattr(detector, "changepoint_probs", None)
        has_probs = cps_probs is not None and len(cps_probs) > 0

        for i, cp in enumerate(sorted(change_points), 1):
            if 0 <= cp < len(dates):
                prob_str = "N/A"
                if has_probs:
                    ridx = cp - 1
                    if 0 <= ridx < len(cps_probs):
                        prob_str = f"{float(cps_probs[ridx]):.3f}"
                out.append(f"    变点{i:2d}: {dates[cp].strftime('%Y-%m-%d')}, "
                           f"价格={data[cp]:.2f}, R[0]={prob_str}")

        # 变点间隔（原版逻辑：>=2 才显示）
        if len(change_points) >= 2:
            intervals = np.diff(sorted(change_points))
            out.append("\n【变点间隔】")
            out.append(f"  平均: {np.mean(intervals):.1f} 周")
            out.append(f"  最短: {np.min(intervals):.1f} 周")
            out.append(f"  最长: {np.max(intervals):.1f} 周")
            out.append(f"  标准差: {np.std(intervals):.1f} 周")

        # 分段分析（恢复原版逻辑：有变点才显示整个段落）
        if change_points:
            out.append("\n【分段分析】")
            for i, (s, e) in enumerate(split_segments(len(data), change_points), 1):
                if e > s:
                    seg = data[s:e + 1]
                    ret = (seg[-1] - seg[0]) / seg[0] * 100
                    with np.errstate(divide='ignore', invalid='ignore'):
                        diff = np.diff(seg) / seg[:-1]
                    vol = np.nanstd(diff) * 100 if len(diff) > 0 else 0.0
                    trend = ("📈 强劲上涨" if ret > 10 else
                             "↗️ 温和上涨" if ret > 0 else
                             "↘️ 温和下跌" if ret > -10 else "📉 大幅下跌")
                    out.append(f"  阶段{i}: {dates[s].strftime('%Y-%m-%d')} ~ "
                               f"{dates[e].strftime('%Y-%m-%d')} | "
                               f"{e - s + 1}周 | 收益 {ret:+.2f}% | "
                               f"波动 {vol:.2f}% | {trend}")

        # 在线预警（恢复原版逻辑：有数据且 warn 非空才显示整段）
        if has_probs:
            warn = detector.recent_warning(cps_probs, window=8)
            if warn:
                out.append("\n【在线预警（最近 8 期）】")
                T = len(cps_probs)
                w = warn['window']
                for i, p in enumerate(warn['recent_cp_probs']):
                    ridx = T - w + i
                    if 0 <= ridx + 1 < len(dates):
                        d = dates[ridx + 1].strftime('%Y-%m-%d')
                        flag = " ⚠" if p > config.cp_threshold else ""
                        out.append(f"  {d}: R[0] = {p:.4f}{flag}")
                out.append(f"  当前变点概率: {warn['current_cp_prob']:.4f}")
                map_rl_last = safe_last(getattr(detector, "map_run_lengths", None), default=None)
                if map_rl_last is not None:
                    out.append(f"  当前 MAP 运行长度: {int(map_rl_last)} 周")
                else:
                    out.append("  当前 MAP 运行长度: N/A")
                if warn['in_new_segment']:
                    out.append(f"  ⚠ 当前可能处于新段（R[0] > {config.cp_threshold}）")
                else:
                    out.append(f"  ✓ 当前未检测到新变点")

        # 参数
        out.append("\n【BOCPD 参数】")
        out.append(f"  Hazard H: {config.hazard:.6f}  (平均段长 λ = {1/config.hazard:.1f} 周)")
        out.append(f"  变点概率阈值: {config.cp_threshold}")
        out.append(f"  最小变点间隔: {config.min_gap} 周")
        if getattr(detector, "prior", None) is not None:
            p = detector.prior
            out.append(f"  NIG 先验: μ0={p.mu0:.6f}, κ0={p.kappa0:.4f}, "
                       f"α0={p.alpha0:.4f}, β0={p.beta0:.6e}")
        out.append(f"  最大运行长度: {config.max_run_length}")
        out.append("=" * 60)
        return "\n".join(out)


# ==================== 6. 分析主流程 ====================
class ChangePointAnalysis:
    def __init__(self, config=None, log_callback=None):
        self.config = config or Config()
        self.log_callback = log_callback or print
        self.data = None
        self.dates = None
        self.returns = None
        self.change_points = None
        self.detector = None
        self.figure_paths = []

    def _log(self, msg):
        self.log_callback(msg)

    def run_analysis(self):
        self._log("=" * 60)
        self._log(f"开始 BOCPD 分析: {self.config.stock_name} ({self.config.stock_code})")
        self._log(f"频率={self.config.frequency}, {self.config.start_date} ~ {self.config.end_date}")
        self._log("=" * 60)

        os.makedirs(self.config.output_dir, exist_ok=True)

        df = DataFetcher.get_stock_data(self.config, log_callback=self._log)
        if df is None or len(df) == 0:
            self._log(f"未获取到 {self.config.stock_code} 数据")
            return None

        self.data, self.dates = DataFetcher.prepare_data(df)
        if len(self.data) < 10:
            self._log("数据太短，BOCPD 难以可靠检测")
            return None

        self._log(f"数据长度: {len(self.data)} 周, "
                  f"时间: {self.dates[0].date()} ~ {self.dates[-1].date()}")

        self.returns = np.log(self.data[1:] / self.data[:-1])

        self._log("初始化 BOCPD 检测器...")
        self.detector = BOCPDDetector(self.config, log_callback=self._log)

        self._log("开始 BOCPD 回溯检测（在线递推重放）...")
        detected_ret = self.detector.fit(self.returns)
        self._log(f"收益率序列变点数: {len(detected_ret)}")

        self.change_points = [cp + 1 for cp in detected_ret if cp + 1 < len(self.data)]
        self._log(f"价格序列变点数: {len(self.change_points)}")

        # 【恢复原格式】0 变点时只给一行简短日志，不列"可尝试..."建议
        if len(self.change_points) == 0:
            self._log("未检测到变点")

        # 在线预警日志（有数据时打印）
        cps_probs = self.detector.changepoint_probs
        if cps_probs is not None and len(cps_probs) > 0:
            warn = self.detector.recent_warning(cps_probs, window=8)
            if warn:
                self._log(f"当前变点概率: {warn['current_cp_prob']:.4f}")
                map_rl_last = safe_last(self.detector.map_run_lengths, default=None)
                if map_rl_last is not None:
                    self._log(f"当前 MAP 运行长度: {int(map_rl_last)} 周")
                if warn['in_new_segment']:
                    self._log(f"⚠ 当前可能处于新段（R[0] > {self.config.cp_threshold}）")
                else:
                    self._log("✓ 当前未检测到新变点")

        # 图表
        if self.config.save_figures:
            self._log("生成图表...")
            try:
                fig1 = Visualizer.plot_comprehensive_analysis(
                    self.data, self.dates, self.change_points,
                    self.detector.changepoint_probs,
                    self.detector.map_run_lengths, self.config)
                p1 = os.path.join(self.config.output_dir,
                                  f"{self.config.stock_code}_bocpd_analysis.png")
                fig1.savefig(p1, dpi=self.config.dpi, bbox_inches='tight')
                plt.close(fig1)

                fig2 = Visualizer.plot_simple_chart(
                    self.data, self.dates, self.change_points, self.config)
                p2 = os.path.join(self.config.output_dir,
                                  f"{self.config.stock_code}_bocpd_simple.png")
                fig2.savefig(p2, dpi=self.config.dpi, bbox_inches='tight')
                plt.close(fig2)

                self.figure_paths = [p1, p2]
                self._log(f"图表: {p1}")
                self._log(f"图表: {p2}")
            except Exception as e:
                self._log(f"生成图表失败: {e}")
                traceback.print_exc()

        # 结果 CSV
        if self.config.save_results:
            try:
                probs = self.detector.changepoint_probs
                if probs is not None and len(probs) > 0:
                    n_ret = len(probs)
                    prob_df = pd.DataFrame({
                        'date': self.dates[1:1 + n_ret],
                        'posterior_prob': probs,
                        'map_run_length': self.detector.map_run_lengths[:n_ret],
                    })
                    prob_path = os.path.join(
                        self.config.output_dir,
                        f"{self.config.stock_code}_bocpd_probs.csv")
                    prob_df.to_csv(prob_path, index=False)

                res_df = pd.DataFrame({
                    'date': self.dates,
                    'close': self.data,
                    'is_changepoint': False,
                })
                for cp in self.change_points:
                    if 0 <= cp < len(res_df):
                        res_df.loc[cp, 'is_changepoint'] = True
                res_path = os.path.join(
                    self.config.output_dir,
                    f"{self.config.stock_code}_bocpd_results.csv")
                res_df.to_csv(res_path, index=False)
                self._log(f"结果 CSV: {res_path}")
            except Exception as e:
                self._log(f"保存结果失败: {e}")
                traceback.print_exc()

        # 报告
        if self.config.generate_report:
            try:
                report = ReportGenerator.generate_report(
                    self.data, self.dates, self.change_points,
                    self.detector, self.config)
                rp = os.path.join(self.config.output_dir,
                                  f"{self.config.stock_code}_bocpd_report.txt")
                with open(rp, 'w', encoding='utf-8') as f:
                    f.write(report)
                self._log(f"报告: {rp}")
            except Exception as e:
                self._log(f"保存报告失败: {e}")
                traceback.print_exc()

        self._log("分析完成。")
        return self

    def build_report(self):
        if self.data is None or self.detector is None:
            return "无结果"
        try:
            return ReportGenerator.generate_report(
                self.data, self.dates, self.change_points,
                self.detector, self.config)
        except Exception as e:
            return f"报告生成失败: {e}\n\n{traceback.format_exc()}"


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
        self.root.title("BOCPD 在线贝叶斯变点检测系统 - 周线")
        self.root.geometry("980x780")

        self.log_queue = queue.Queue()
        self.worker = None

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        top = ttk.LabelFrame(self.root, text="参数设置", padding=8)
        top.pack(fill='x', padx=8, pady=6)

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

        r2 = ttk.Frame(top)
        r2.pack(fill='x', pady=2)
        ttk.Label(r2, text="开始日期:").pack(side='left')
        self.start_var = tk.StringVar(value="2015-01-01")
        ttk.Entry(r2, textvariable=self.start_var, width=14).pack(side='left', padx=4)
        ttk.Label(r2, text="结束日期:").pack(side='left', padx=(16, 0))
        self.end_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        ttk.Entry(r2, textvariable=self.end_var, width=14).pack(side='left', padx=4)

        self.advanced_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="显示高级参数",
                        variable=self.advanced_visible,
                        command=self._toggle_advanced).pack(anchor='w', pady=(6, 0))

        self.advanced_frame = ttk.Frame(top)
        adv = self.advanced_frame
        row = ttk.Frame(adv)
        row.pack(fill='x', pady=2)
        ttk.Label(row, text="hazard H:").pack(side='left')
        self.hazard_var = tk.StringVar(value="0.02")
        ttk.Entry(row, textvariable=self.hazard_var, width=8).pack(side='left', padx=(2, 14))
        ttk.Label(row, text="阈值 p_c:").pack(side='left')
        self.pc_var = tk.StringVar(value="0.5")
        ttk.Entry(row, textvariable=self.pc_var, width=6).pack(side='left', padx=(2, 14))
        ttk.Label(row, text="min_gap:").pack(side='left')
        self.mingap_var = tk.StringVar(value="8")
        ttk.Entry(row, textvariable=self.mingap_var, width=6).pack(side='left', padx=(2, 14))

        row2 = ttk.Frame(adv)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="NIG κ0:").pack(side='left')
        self.k0_var = tk.StringVar(value="1.0")
        ttk.Entry(row2, textvariable=self.k0_var, width=6).pack(side='left', padx=(2, 14))
        ttk.Label(row2, text="NIG α0:").pack(side='left')
        self.a0_var = tk.StringVar(value="1.0")
        ttk.Entry(row2, textvariable=self.a0_var, width=6).pack(side='left', padx=(2, 14))
        ttk.Label(row2, text="max_run_length:").pack(side='left')
        self.maxrl_var = tk.StringVar(value="5000")
        ttk.Entry(row2, textvariable=self.maxrl_var, width=8).pack(side='left', padx=(2, 14))

        bar = ttk.Frame(self.root)
        bar.pack(fill='x', padx=8, pady=4)
        self.run_btn = ttk.Button(bar, text="开始分析", command=self._start_analysis)
        self.run_btn.pack(side='left')
        ttk.Button(bar, text="清空日志", command=self._clear_log).pack(side='left', padx=6)
        ttk.Button(bar, text="打开输出目录", command=self._open_output).pack(side='left')

        self.progress = ttk.Progressbar(bar, mode='indeterminate', length=200)
        self.progress.pack(side='right')

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

        online_tab = ttk.Frame(self.notebook)
        self.notebook.add(online_tab, text="在线监控")
        self.online_text = scrolledtext.ScrolledText(online_tab, wrap='word',
                                                     font=('Consolas', 10))
        self.online_text.pack(fill='both', expand=True)

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
                config.hazard = float(self.hazard_var.get())
                config.cp_threshold = float(self.pc_var.get())
                config.min_gap = int(self.mingap_var.get())
                config.nig_kappa0 = float(self.k0_var.get())
                config.nig_alpha0 = float(self.a0_var.get())
                config.max_run_length = int(self.maxrl_var.get())
                if not (0 < config.hazard < 1):
                    raise ValueError("hazard H 必须在 (0,1) 内")
            except ValueError as e:
                messagebox.showwarning("提示", f"高级参数格式错误: {e}")
                return

        config.verbose = True

        self.run_btn.config(state='disabled')
        self.progress.start(10)
        self.status_var.set("分析中...")
        self._append_log("\n" + "#" * 60)
        self._append_log(f"启动 BOCPD 分析: {config.stock_name} ({config.stock_code})")

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
            self.notebook.select(1)
        except Exception as e:
            self._append_log(f"报告生成失败: {e}")

        # 在线监控摘要（恢复原格式：有数据正常显示，无数据简短提示）
        try:
            detector = analyzer.detector
            cps = getattr(detector, "changepoint_probs", None) if detector else None
            if detector is None or cps is None or len(cps) == 0:
                self.online_text.delete('1.0', 'end')
                self.online_text.insert('end', "【在线监控】\n  暂无数据")
            else:
                warn = detector.recent_warning(cps, window=8)
                if not warn:
                    self.online_text.delete('1.0', 'end')
                    self.online_text.insert('end', "【在线监控】\n  暂无数据")
                else:
                    lines = ["【在线监控 - 最近 8 期】\n"]
                    T = len(cps)
                    w = warn['window']
                    for i, p in enumerate(warn['recent_cp_probs']):
                        ridx = T - w + i
                        if 0 <= ridx + 1 < len(analyzer.dates):
                            d = analyzer.dates[ridx + 1].strftime('%Y-%m-%d')
                            flag = " ⚠" if p > analyzer.config.cp_threshold else ""
                            lines.append(f"  {d}: R[0] = {p:.4f}{flag}")
                    lines.append(f"\n当前变点概率: {warn['current_cp_prob']:.4f}")

                    map_rl_last = safe_last(
                        getattr(detector, "map_run_lengths", None), default=None)
                    if map_rl_last is not None:
                        lines.append(f"当前 MAP 运行长度: {int(map_rl_last)} 周")
                    else:
                        lines.append("当前 MAP 运行长度: N/A")

                    lines.append(f"当前状态: "
                                 f"{'⚠ 可能处于新段' if warn['in_new_segment'] else '✓ 平稳'}")

                    lines.append("\n注：BOCPD 是天然在线算法。")
                    lines.append("    新数据到达时，调用 detector.online_step(x)")
                    lines.append("    即可得到最新 R[0] 和 MAP 运行长度，")
                    lines.append("    无需重算历史。")
                    self.online_text.delete('1.0', 'end')
                    self.online_text.insert('end', "\n".join(lines))
        except Exception as e:
            self._append_log(f"在线监控摘要生成失败: {e}")
            traceback.print_exc()

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