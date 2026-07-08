"""
模型对比实验: Attention-LSTM vs 普通 LSTM (PyTorch)

对 normalizeResult.mat 的 SSA 分解结果，对比以下几种模型的时序预测性能:
  - 纯 LSTM 基线
  - PyTorch MultiheadAttention 模型
  - PyTorch Attention LSTM (来自 model/attention_lstm_torch)

输出: results/Only_LSTM_rmse.csv, results/Attenrion_LSTM_rmse.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ── 项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.attention_lstm_torch import AttentionLSTM

RESULTS_DIR = PROJECT_ROOT / 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

NORMALIZE_MAT = PROJECT_ROOT / 'normalizeResult.mat'

# ── 设备 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 模型定义 ──

class PlainLSTM(nn.Module):
    """纯 LSTM 基线"""
    def __init__(self, input_dim, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class MultiheadAttentionModel(nn.Module):
    """使用 PyTorch MultiheadAttention 的自注意力模型"""
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out[:, -1, :]  # 取最后时间步


# ── 工具函数 ──

def split_data(data_x, data_y, percent):
    """按比例划分数据集"""
    return data_x[:int(percent * len(data_y))], data_y[:int(percent * len(data_y))]


def predict_torch(model, valid_x, valid_y, csv_path):
    """PyTorch 模型：逐样本预测，记录 RMSE 并绘图"""
    full_path = RESULTS_DIR / csv_path
    if os.path.exists(full_path):
        os.remove(full_path)

    model.eval()
    sum_rmse = 0
    predict_result = []

    for X, y in zip(valid_x, valid_y):
        X_t = torch.from_numpy(X.reshape(1, X.shape[0], X.shape[1])).float().to(device)
        with torch.no_grad():
            y_pred = model(X_t).cpu().numpy()[0]
        predict_result.append(y_pred)
        rmse = np.sqrt(mean_squared_error(y.reshape(1, -1), y_pred.reshape(1, -1)))
        sum_rmse += rmse
        with open(full_path, 'a') as f:
            f.write(f'rmse,{rmse}\n')

    rows = len(predict_result)
    plt.plot(range(rows), [row[0] for row in predict_result], label='predict')
    plt.plot(range(rows), [row[0] for row in valid_y], label='valid')
    plt.xlabel('time - axis')
    plt.ylabel('A - axis')
    plt.title(csv_path)
    plt.legend()
    plt.show()
    return sum_rmse


def train_model(model, train_x, train_y, epochs, batch_size=32):
    """通用 PyTorch 训练循环"""
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    x_t = torch.from_numpy(train_x).float().to(device)
    y_t = torch.from_numpy(train_y).float().to(device)

    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.6f}")

    return model


# ── 实验函数 ──

def attention_lstm(X, y):
    """MultiheadAttention 模型实验"""
    train_x, train_y = split_data(X, y, 0.8)
    valid_x, valid_y = X[int(0.8 * len(y)):], y[int(0.8 * len(y)):]

    model = MultiheadAttentionModel(input_dim=train_x.shape[2]).to(device)
    print("MultiheadAttention model summary:")
    print(model)
    model = train_model(model, train_x, train_y, epochs=50)
    return predict_torch(model, valid_x, valid_y, 'Attenrion_LSTM_rmse.csv')


def only_lstm(X, y):
    """纯 LSTM 基线实验"""
    train_x, train_y = X[:int(0.8 * len(y))], y[:int(0.8 * len(y))]
    valid_x, valid_y = X[int(0.8 * len(y)):], y[int(0.8 * len(y)):]

    model = PlainLSTM(input_dim=train_x.shape[2]).to(device)
    print("Plain LSTM summary:")
    print(model)
    model = train_model(model, train_x, train_y, epochs=20)
    return predict_torch(model, valid_x, valid_y, 'Only_LSTM_rmse.csv')


def attention2(X, y):
    """PyTorch Attention LSTM 实验 (model/attention_lstm_torch.py)"""
    train_x = torch.from_numpy(X[:int(0.8 * len(y))]).float().to(device)
    train_y = torch.from_numpy(y[:int(0.8 * len(y))]).float().to(device)

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    model = AttentionLSTM(input_size=X.shape[2], hidden_size=10, output_size=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for i, (inputs, labels) in enumerate(loader):
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            if i % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, 10, i + 1, len(loader), loss.item()))

    valid_x = torch.from_numpy(X[int(0.8 * len(y)):]).float().to(device)
    valid_y = torch.from_numpy(y[int(0.8 * len(y)):]).float().to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(valid_x)
        loss = criterion(outputs, valid_y)
    print('Attention2 MSE on validation set: {:.4f}'.format(loss.item()))
    return loss.item()


# ── 数据加载 ──

def data_loader(path):
    data = loadmat(str(path))
    results = np.array(data['ssa_results'])
    n_src = results['n_src'][0][0]
    origin_data = n_src[3000:4000, :]
    return split_sequences(origin_data, 12)


def split_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        X.append(data[i:end_ix])
        y.append(data[end_ix])
    return np.array(X), np.array(y)


# ── 主入口 ──

if __name__ == "__main__":
    X, y = data_loader(NORMALIZE_MAT)
    print("attention-lstm mse:")
    print(attention_lstm(X, y))
    print("only-lstm mse:")
    print(only_lstm(X, y))
