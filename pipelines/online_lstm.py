"""
SSA-LSTM 在线增量学习管线 — PyTorch 版本

对经 MATLAB SSA 工具预分解的数据 (normalizeResult.mat) 进行在线增量 LSTM 建模。
通过 RMSE 阈值检测工况变化，自动触发模型重建，实现多工况自适应时序预测。

核心机制:
  - 初始建模: 取前 200 个样本训练 LSTM
  - 逐样本预测: 每步计算 RMSE
  - 阈值判断: RMSE > 1.5 则检测连续异常
  - 增量微调: 每积累 32 个正常样本微调模型
  - 递归重建: 检测到模态变化后保存模型并重建

数据来源: normalizeResult.mat (由外部 SSA 工具生成)
"""
import os
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error

# ── 确定性 ──
torch.manual_seed(11)

# ── 项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / 'results'
SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models'
NORMALIZE_MAT = PROJECT_ROOT / 'normalizeResult.mat'
LOSS_CSV = RESULTS_DIR / 'loss.csv'
NON_NOTE_CSV = RESULTS_DIR / 'non-note.csv'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ── 设备 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class OnlineLSTMModel(nn.Module):
    """在线增量 LSTM 单步预测模型"""

    def __init__(self, input_dim, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        return self.fc(out[:, -1, :])  # 最后时间步 → (batch, input_dim)


def data_pre_deal(path):
    """加载 MATLAB SSA 工具生成的分解结果"""
    data = loadmat(str(path))
    results = np.array(data['ssa_results'])
    n_src = results['n_src'][0][0]
    return n_src


def split_sequences(data, n_steps):
    """将时序数据切分为输入-输出序列对"""
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        X.append(data[i:end_ix])
        y.append(data[end_ix])
    return np.array(X), np.array(y)


def _train_batch(model, x, y, epochs, optimizer, criterion):
    """训练模型若干 epoch"""
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def _predict_one(model, x):
    """单样本预测"""
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


def judge_error(model, x_test, y_test, threshold, n):
    """递归检查后续 n 个样本是否全部超阈值"""
    if n <= 0:
        return n
    X = x_test[-n].reshape((1, x_test.shape[1], x_test.shape[2]))
    y = y_test[-n].reshape((1, y_test.shape[1]))
    y_pred = _predict_one(model, torch.from_numpy(X).float().to(device))
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    if rmse > threshold:
        return judge_error(model, x_test, y_test, threshold, n - 1)
    else:
        return -1


def model_train_prediction(X_all, y_all, start, windows, n_steps):
    """在线增量学习 + 模态切换检测的核心循环"""
    n_features = X_all.shape[2]

    # 初始训练集
    init_X_train = torch.from_numpy(X_all[start:start + windows]).float().to(device)
    init_y_train = torch.from_numpy(y_all[start:start + windows]).float().to(device)
    init_X_test = X_all[start + windows:]
    init_y_test = y_all[start + windows:]
    start = start + windows

    # 创建并训练初始模型
    model = OnlineLSTMModel(input_dim=n_features, hidden_size=50).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    _train_batch(model, init_X_train, init_y_train, epochs=50, optimizer=optimizer, criterion=criterion)

    threshold = 1.5
    addX_list, addy_list = [], []

    for X, y in zip(init_X_test, init_y_test):
        X_batch = torch.from_numpy(X.reshape(1, X.shape[0], X.shape[1])).float().to(device)
        y_batch = y.reshape(1, y.shape[0])

        y_pred = _predict_one(model, X_batch)
        rmse = np.sqrt(mean_squared_error(y_batch, y_pred))

        # 记录 RMSE
        with open(LOSS_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([start, rmse])

        # 阈值检测
        if rmse > threshold:
            X_check = X_all[start + 1:start + 3]
            y_check = y_all[start + 1:start + 3]
            if judge_error(model, X_check, y_check, threshold, len(y_check) - 1) == 0:
                # 模态变化确认 → 保存模型并递归重建
                torch.save(model.state_dict(),
                           str(SAVED_MODELS_DIR / ('lstm_model_' + str(start) + '.pt')))
                with open(NON_NOTE_CSV, 'a', newline='') as f:
                    csv.writer(f).writerow([start, rmse])
                    print("non-note is save!")
                model_train_prediction(X_all, y_all, start, windows, n_steps)
                break

        # 积累正常样本
        addX_list.append(X_batch)
        addy_list.append(y_batch)
        start = start + 1

        # 增量微调
        if len(addy_list) >= 32:
            addX_t = torch.cat(addX_list[-32:], dim=0)
            addy_t = torch.from_numpy(np.array(addy_list[-32:])).float().to(device)
            _train_batch(model, addX_t, addy_t, epochs=50, optimizer=optimizer, criterion=criterion)
            addX_list, addy_list = [], []

    # 到达数据末尾，保存最后一个模型
    if start == len(y_all) + n_steps - 1:
        torch.save(model.state_dict(),
                   str(SAVED_MODELS_DIR / ('lstm_model_' + str(start) + '.pt')))


def recognize_module():
    """预留: 模态识别接口 (待实现)"""
    pass


if __name__ == "__main__":
    print("GPU available:", torch.cuda.is_available())
    mat_data = data_pre_deal(NORMALIZE_MAT)
    X_all, y_all = split_sequences(mat_data, 12)

    if os.path.exists(LOSS_CSV):
        os.remove(LOSS_CSV)

    model_train_prediction(X_all, y_all, 0, 200, 12)
