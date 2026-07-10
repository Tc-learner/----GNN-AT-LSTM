"""
自适应阈值检测器

支持两种模式:
  - KDEAdaptiveThreshold: 标量RMSE的全局KDE (原有)
  - NodeKDEThreshold:       节点级KDE联合判断 (新增, 用于GNN模型)
"""
import numpy as np
from scipy import stats
from collections import deque


class KDEAdaptiveThreshold:
    """基于核密度估计的标量自适应阈值检测器 (用于LSTM等标量RMSE场景)"""

    def __init__(self, confidence=0.99, window_size=200, confirm_steps=3,
                 bw_method='scott'):
        self.confidence = confidence
        self.window_size = window_size
        self.confirm_steps = confirm_steps
        self.bw_method = bw_method
        self.kde = None
        self.threshold = None
        self.threshold_low = None
        self.threshold_high = None
        self.residual_buffer = deque(maxlen=window_size)
        self.initialized = False
        self.alarm_count = 0
        self.is_alarm = False

    def add_residual(self, residual):
        self.residual_buffer.append(residual)
        if self.initialized:
            if self.threshold_high is not None and (residual > self.threshold_high or residual < self.threshold_low):
                self.alarm_count += 1
            else:
                self.alarm_count = 0
            self.is_alarm = self.alarm_count >= self.confirm_steps

    def fit_kde(self):
        if len(self.residual_buffer) < 10:
            self.threshold = 1.5
            self.threshold_low = -1.5
            self.threshold_high = 1.5
            self.initialized = True
            return self.threshold
        residuals = np.array(list(self.residual_buffer))
        try:
            self.kde = stats.gaussian_kde(residuals, bw_method=self.bw_method)
            x_min, x_max = residuals.min() - 3*residuals.std(), residuals.max() + 3*residuals.std()
            x_grid = np.linspace(x_min, x_max, 1000)
            pdf_vals = self.kde.evaluate(x_grid)
            cdf_vals = np.cumsum(pdf_vals)
            cdf_vals /= cdf_vals[-1]
            alpha = 1.0 - self.confidence
            low_idx = np.searchsorted(cdf_vals, alpha/2)
            high_idx = np.searchsorted(cdf_vals, 1.0 - alpha/2)
            self.threshold_low = float(x_grid[low_idx])
            self.threshold_high = float(x_grid[min(high_idx, len(x_grid)-1)])
            self.threshold = max(abs(self.threshold_low), abs(self.threshold_high))
        except (np.linalg.LinAlgError, ValueError):
            mu, sigma = residuals.mean(), residuals.std()
            z = stats.norm.ppf(1.0 - (1.0 - self.confidence)/2)
            self.threshold_low, self.threshold_high = mu - z*sigma, mu + z*sigma
            self.threshold = max(abs(self.threshold_low), abs(self.threshold_high))
        self.alarm_count = 0
        self.is_alarm = False
        self.initialized = True
        return self.threshold

    def reset(self, keep_buffer=False):
        self.alarm_count = 0; self.is_alarm = False; self.initialized = False
        self.kde = None; self.threshold = None
        if not keep_buffer: self.residual_buffer.clear()


class NodeKDEThreshold:
    """
    节点级 KDE 联合判断检测器 (用于 GNN 等多维输出场景)

    对每个设备节点的预测残差分别建立独立KDE模型。
    综合判断策略: 当 ≥ vote_threshold 个节点同时报警时, 触发全局报警。

    Args:
        n_nodes:          节点数 (default: 8)
        confidence:       置信水平 (default: 0.99)
        window_size:      KDE 拟合窗口 (default: 200)
        confirm_steps:    连续确认步数 (default: 3)
        vote_threshold:   触发报警的最少报警节点数 (default: n_nodes//2+1, 即多数)
        strategy:         综合策略: 'majority' | 'any' | 'max'
    """

    def __init__(self, n_nodes=8, confidence=0.99, window_size=200,
                 confirm_steps=3, vote_threshold=None, strategy='majority'):
        self.n_nodes = n_nodes
        self.confidence = confidence
        self.window_size = window_size
        self.confirm_steps = confirm_steps
        self.strategy = strategy
        self.vote_threshold = vote_threshold or max(1, n_nodes // 2 + 1)

        # 每个节点一个独立的 KDE
        self.node_detectors = [
            KDEAdaptiveThreshold(confidence=confidence, window_size=window_size,
                                 confirm_steps=confirm_steps)
            for _ in range(n_nodes)
        ]
        self.initialized = False
        self.is_alarm = False
        self.alarm_count = 0
        self.node_alarm_state = np.zeros(n_nodes, dtype=bool)

    def add_residuals(self, residuals):
        """
        添加一个多维残差向量。

        Args:
            residuals: (n_nodes,) 每个节点的预测残差
        """
        for i, det in enumerate(self.node_detectors):
            det.add_residual(float(residuals[i]))

        # 检查各节点是否初始化完成
        if not self.initialized and all(d.initialized for d in self.node_detectors):
            self.initialized = True

        # 综合判断
        self.node_alarm_state = np.array([d.is_alarm for d in self.node_detectors])
        n_alarmed = self.node_alarm_state.sum()

        if self.strategy == 'majority':
            is_alarm_now = n_alarmed >= self.vote_threshold
        elif self.strategy == 'any':
            is_alarm_now = n_alarmed >= 1
        elif self.strategy == 'max':
            is_alarm_now = n_alarmed >= self.n_nodes
        else:
            is_alarm_now = n_alarmed >= self.vote_threshold

        if is_alarm_now:
            self.alarm_count += 1
        else:
            self.alarm_count = 0

        self.is_alarm = self.alarm_count >= self.confirm_steps

    def fit_kde(self, residuals_buffer):
        """
        用残差缓冲区分拟合所有节点的KDE。

        Args:
            residuals_buffer: list of (n_nodes,) arrays
        """
        if len(residuals_buffer) < 10:
            return
        residuals_arr = np.array(list(residuals_buffer))
        for i, det in enumerate(self.node_detectors):
            for r in residuals_arr[:, i]:
                det.add_residual(r)
            det.fit_kde()

    def reset(self):
        self.alarm_count = 0; self.is_alarm = False
        for d in self.node_detectors: d.reset()

    def get_thresholds(self):
        return np.array([d.threshold for d in self.node_detectors])

    def get_node_alarm_state(self):
        return self.node_alarm_state.copy()


def create_adaptive_threshold(confidence=0.99, window_size=200, confirm_steps=3):
    return KDEAdaptiveThreshold(confidence=confidence, window_size=window_size,
                                confirm_steps=confirm_steps)


def create_node_kde_threshold(n_nodes=8, confidence=0.99, window_size=200,
                               confirm_steps=3, strategy='majority'):
    return NodeKDEThreshold(n_nodes=n_nodes, confidence=confidence,
                             window_size=window_size, confirm_steps=confirm_steps,
                             strategy=strategy)


# ── 自检 ──
if __name__ == "__main__":
    import time
    print("Testing NodeKDEThreshold...")

    np.random.seed(42)
    dt = NodeKDEThreshold(n_nodes=8, confidence=0.99, window_size=200, strategy='majority')

    # 初始化: 200个正常样本
    for _ in range(200):
        dt.add_residuals(np.random.normal(0.3, 0.1, 8))
        if not dt.initialized and all(len(d.residual_buffer) >= 200 for d in dt.node_detectors):
            for d in dt.node_detectors:
                d.fit_kde()

    # 注入异常: 3个节点大幅偏离
    n_alarms = 0
    for _ in range(100):
        r = np.random.normal(0.3, 0.1, 8)
        # 随机让2个节点偏离
        r[np.random.choice(8, 2, replace=False)] += np.random.uniform(1.5, 2.5, 2)
        dt.add_residuals(r)
        if dt.is_alarm: n_alarms += 1

    print(f"Anomaly detection: {n_alarms}/100 alarms triggered")
    print(f"Node thresholds: {dt.get_thresholds()}")
    print(f"Strategy: majority (need {dt.vote_threshold}/8 nodes)")
