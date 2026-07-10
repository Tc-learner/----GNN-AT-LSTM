"""
GNN-AT-LSTM 实验管线 — 完整的训练、检测、评估流程

基于 SSA-LSTM 在线增量学习框架，整合图神经网络空间编码和自适应阈值检测。

运行:
    python pipelines/gnn_at_lstm_pipeline.py

实验模式:
  --mode compare : 对比 GNN-AT-LSTM vs SSA-LSTM vs 其它基线 (默认)
  --mode train   : 仅训练 GNN-AT-LSTM 模型
  --mode ablate  : 消融实验 (GNN vs 无GNN, KDE vs 固定阈值)
"""
import os
import csv
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.process_graph import (
    build_process_graph, sensors_to_node_features,
    NUM_NODES, NODE_NAMES,
)
from model.gnn_at_lstm import GNNATLSTM
from model.adaptive_threshold import KDEAdaptiveThreshold

# ── 路径配置 ──
RESULTS_DIR = PROJECT_ROOT / 'results'
SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ── 设备 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 确定性 ──
torch.manual_seed(11)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_te_data(mat_path):
    """加载 TE 过程 .mat 数据，返回 (n_timesteps, 52) XMEAS 数组"""
    data = loadmat(str(mat_path))
    # 原始 TE 数据 (simout = 52列 XMEAS)
    if 'simout' in data:
        arr = data['simout']
        if arr.shape[1] >= 52:
            return arr[:, :52]
        return arr
    # SSA 分解后的数据 — 列数不定，直接返回
    elif 'ssa_results' in data:
        results = np.array(data['ssa_results'])
        n_src = results['n_src'][0][0]
        return n_src
    else:
        raise KeyError(f"Unknown data format in {mat_path}: keys={list(data.keys())}")


def split_sequences(data, n_steps):
    """滑动窗口切分"""
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        X.append(data[i:end_ix])
        y.append(data[end_ix])
    return np.array(X), np.array(y)


# ═══════════════════════════════════════════════════════════════
# 训练与评估
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, x_train, y_train, optimizer, criterion, batch_size=32):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    n_batches = max(1, len(x_train) // batch_size)
    indices = np.random.permutation(len(x_train))
    use_graph = getattr(model, 'use_graph', True)

    for i in range(n_batches):
        batch_idx = indices[i * batch_size:(i + 1) * batch_size]
        bx = torch.from_numpy(x_train[batch_idx]).float().to(device)
        by = torch.from_numpy(y_train[batch_idx]).float().to(device)

        optimizer.zero_grad()
        pred = model(bx)

        # 目标: 图模式 → 转换为节点特征; 非图模式 → 直接使用原始传感器值
        if use_graph:
            by_target = sensors_to_node_features(by)
        else:
            by_target = by  # 直接预测原始传感器值

        loss = criterion(pred, by_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_batches


def predict_batch(model, x):
    """批量预测"""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).float().to(device)
        # 如果 x 是单样本，加 batch 维
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)
        pred = model(x_t)
        return pred.cpu().numpy()


def evaluate_detector(model, detector, X, y, include_attention=False):
    """
    逐样本评估检测器性能。

    Returns:
        dict: {
            'rmse_list': [...],
            'alarms': [...],
            'detected_steps': [...],
            'fault_scores': [...] (per-node residuals for localization),
        }
    """
    results = {'rmse_list': [], 'alarms': [], 'detected_steps': [], 'fault_scores': []}
    detector.reset(keep_buffer=False)

    for i, (x_i, y_i) in enumerate(zip(X, y)):
        x_batch = x_i.reshape(1, *x_i.shape)
        pred = predict_batch(model, x_batch)[0]  # (output_dim,)
        if getattr(model, 'use_graph', True):
            actual = sensors_to_node_features(y_i.reshape(1, -1))[0]
        else:
            actual = y_i

        rmse = np.sqrt(mean_squared_error(actual.reshape(-1), pred.reshape(-1)))

        # 自适应阈值检测
        if detector.initialized:
            detector.add_residual(rmse)
            if detector.is_alarm:
                results['detected_steps'].append(i)
                results['alarms'].append(1)
            else:
                results['alarms'].append(0)
        else:
            # 初始化阶段: 仅积累残差
            detector.add_residual(rmse)
            results['alarms'].append(0)

            # 积累足够数据后拟合 KDE
            if len(detector.residual_buffer) >= detector.window_size:
                detector.fit_kde()

        results['rmse_list'].append(rmse)

        # 故障定位分数 (各维度残差)
        residuals = np.abs(actual.reshape(-1) - pred.reshape(-1))
        results['fault_scores'].append(residuals)

    return results


# ═══════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════

def run_experiment(data_path, n_steps=12, init_window=200, epochs=50,
                   use_graph=True, use_kde=True, confidence=0.99,
                   fault_start=None):
    """
    运行完整的 GNN-AT-LSTM 实验。

    Args:
        data_path:      .mat 数据路径
        n_steps:        输入序列长度
        init_window:    初始训练窗口大小
        epochs:         训练 epoch 数
        use_graph:      是否启用 GNN 图结构 (消融用)
        use_kde:        是否使用 KDE 自适应阈值
        confidence:     KDE 置信水平
        fault_start:    故障起始位置 (若已知), 用于评估

    Returns:
        dict: 实验结果
    """
    print(f"\n{'='*60}")
    print(f"Experiment: GNN={'ON' if use_graph else 'OFF'}, KDE={'ON' if use_kde else 'OFF'}")
    print(f"Data: {data_path}")
    print(f"{'='*60}")

    # 加载数据
    raw_data = load_te_data(data_path)
    if raw_data.shape[1] > 52:
        raw_data = raw_data[:, :52]  # 只取 XMEAS (前 52 维)
    print(f"Data shape: {raw_data.shape}")

    # 若数据列数不足 52 (非原始传感器数据), 自动禁用图模式
    if raw_data.shape[1] < 52:
        print(f"  Warning: Data has {raw_data.shape[1]} columns (< 52 XMEAS). Disabling graph mode.")
        use_graph = False

    # 构建图 (仅当有 52 列传感器数据时)
    edge_index, edge_weight, edge_type = None, None, None
    if use_graph and raw_data.shape[1] >= 52:
        edge_index, edge_weight, edge_type = build_process_graph(
            data=raw_data[:min(1000, len(raw_data)), :52], use_data_driven=True, n_neighbors=3
        )

    # 标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data)

    # 滑动窗口
    X, y = split_sequences(scaled_data, n_steps)
    print(f"Sequences: X={X.shape}, y={y.shape}")

    # 划分
    train_end = init_window
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]

    # 创建模型
    input_dim = X_train.shape[2]
    model = GNNATLSTM(
        input_dim=input_dim, gat_hidden=64, lstm_hidden=50,
        n_steps=n_steps, use_graph=use_graph,
        output_dim=input_dim,  # 默认输出维度=输入维度
    )
    if use_graph and edge_index is not None:
        model.set_graph(edge_index, edge_weight)
    model = model.to(device)

    # 训练
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    print(f"Training {epochs} epochs on {len(X_train)} samples...")
    t0 = time.time()
    for epoch in range(epochs):
        loss = train_epoch(model, X_train, y_train, optimizer, criterion, batch_size=32)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    # 阈值检测器
    if use_kde:
        detector = KDEAdaptiveThreshold(confidence=confidence, window_size=init_window)
    else:
        # 固定阈值模式
        detector = KDEAdaptiveThreshold(confidence=confidence, window_size=init_window)
        # 将 KDE 替换为固定值 1.5
        detector.fit_kde = lambda: 1.5
        detector.threshold = 1.5
        detector.threshold_low = -1.5
        detector.threshold_high = 1.5
        detector.initialized = True

    # 逐样本检测
    print(f"Running online detection on {len(X_test)} samples...")
    results = evaluate_detector(model, detector, X_test, y_test)

    # 汇总指标
    rmse_mean = np.mean(results['rmse_list'])
    rmse_std = np.std(results['rmse_list'])
    n_alarms = sum(results['alarms'])
    alarm_rate = n_alarms / len(results['alarms']) * 100 if results['alarms'] else 0

    # 若已知故障起始位置，计算检测延迟
    detection_delay = None
    if fault_start is not None and results['detected_steps']:
        post_fault_detections = [s for s in results['detected_steps'] if s >= fault_start - train_end]
        if post_fault_detections:
            detection_delay = post_fault_detections[0] - (fault_start - train_end)

    print(f"\nResults:")
    print(f"  Mean RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"  Alarm rate: {alarm_rate:.1f}%")
    print(f"  Detected at steps: {results['detected_steps'][:10]}...")
    if detection_delay is not None:
        print(f"  Detection delay: {detection_delay} steps")

    return {
        'model': model,
        'detector': detector,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'alarm_rate': alarm_rate,
        'detection_delay': detection_delay,
        'detected_steps': results['detected_steps'],
        'rmse_list': results['rmse_list'],
        'alarms': results['alarms'],
        'train_time': train_time,
    }


# ═══════════════════════════════════════════════════════════════
# 对比实验
# ═══════════════════════════════════════════════════════════════

def run_comparison():
    """运行 GNN vs no-GNN, KDE vs fixed-threshold 的 2×2 对比"""
    data_path = PROJECT_ROOT / 'normalizeResult.mat'
    if not data_path.exists():
        # 回退到 TE 过渡数据
        data_path = PROJECT_ROOT / 'dataSet' / 'TE transition mode data' / 'M1M2XMEAS.mat'
    if not data_path.exists():
        print("No .mat data file found. Please ensure normalizeResult.mat exists.")
        return

    configs = [
        {'use_graph': True,  'use_kde': True,  'label': 'GNN-AT-LSTM (full)'},
        {'use_graph': True,  'use_kde': False, 'label': 'GNN-LSTM (fixed τ)'},
        {'use_graph': False, 'use_kde': True,  'label': 'LSTM + KDE'},
        {'use_graph': False, 'use_kde': False, 'label': 'SSA-LSTM (baseline)'},
    ]

    all_results = []
    for cfg in configs:
        result = run_experiment(data_path, epochs=30, **{k: cfg[k] for k in ['use_graph', 'use_kde']})
        result['label'] = cfg['label']
        all_results.append(result)

    # 汇总对比表
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'RMSE':>8} {'Alarm%':>8} {'Delay':>8} {'Time(s)':>8}")
    print(f"{'-'*70}")
    for r in all_results:
        delay_str = f"{r['detection_delay']}" if r['detection_delay'] is not None else 'N/A'
        print(f"{r['label']:<30} {r['rmse_mean']:>8.4f} {r['alarm_rate']:>8.1f} "
              f"{delay_str:>8} {r['train_time']:>8.1f}")

    # 保存结果
    result_csv = RESULTS_DIR / 'gnn_at_lstm_comparison.csv'
    with open(result_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'RMSE', 'AlarmRate%', 'DetectionDelay', 'TrainTime(s)'])
        for r in all_results:
            writer.writerow([r['label'], r['rmse_mean'], r['alarm_rate'],
                             r['detection_delay'], r['train_time']])
    print(f"\nResults saved to {result_csv}")

    return all_results


def run_ablation():
    """消融实验: GAT 头数、KDE 置信水平、图构建策略"""
    data_path = PROJECT_ROOT / 'normalizeResult.mat'
    if not data_path.exists():
        data_path = PROJECT_ROOT / 'dataSet' / 'TE transition mode data' / 'M1M2XMEAS.mat'
    if not data_path.exists():
        print("No .mat data file found.")
        return

    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    # 1. KDE 置信水平敏感性
    print("\n--- KDE Confidence Sensitivity ---")
    for conf in [0.95, 0.99, 0.999]:
        r = run_experiment(data_path, epochs=20, use_graph=True, use_kde=True, confidence=conf)
        print(f"  Confidence={conf}: RMSE={r['rmse_mean']:.4f}, Alarm%={r['alarm_rate']:.1f}")

    # 2. GAT 头数敏感性
    print("\n--- GAT Heads Sensitivity ---")
    for heads in [1, 2, 4, 8]:
        model = GNNATLSTM(gat_heads=heads, gat_hidden=64, use_graph=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Heads={heads}: Params={n_params:,}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN-AT-LSTM Experiment Pipeline')
    parser.add_argument('--mode', default='compare', choices=['compare', 'train', 'ablate'],
                        help='Experiment mode')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to .mat data file')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--no-graph', action='store_true',
                        help='Disable graph (ablation)')
    parser.add_argument('--no-kde', action='store_true',
                        help='Use fixed threshold (ablation)')
    args = parser.parse_args()

    print(f"GNN-AT-LSTM Pipeline")
    print(f"  Device: {device}")
    print(f"  Mode: {args.mode}")

    if args.mode == 'compare':
        run_comparison()

    elif args.mode == 'train':
        data_path = args.data or (
            PROJECT_ROOT / 'normalizeResult.mat'
            if (PROJECT_ROOT / 'normalizeResult.mat').exists()
            else PROJECT_ROOT / 'dataSet' / 'TE transition mode data' / 'M1M2XMEAS.mat'
        )
        run_experiment(
            Path(data_path),
            epochs=args.epochs,
            use_graph=not args.no_graph,
            use_kde=not args.no_kde,
        )

    elif args.mode == 'ablate':
        run_ablation()

    print("\nDone.")
