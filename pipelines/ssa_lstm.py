"""
SSA + LSTM 时序预测管线 — PyTorch 版本

从 TE 过程 .mat 数据加载原始测量变量，使用 TruncatedSVD 进行
奇异谱分析 (SSA) 分解，训练 LSTM 模型进行单步预测，基于 RMSE
保存最优模型。

数据来源: dataSet/TE transition mode data/M1M2XMEAS.mat
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ── 确定性 ──
torch.manual_seed(11)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── 项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / 'results'
SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ── 设备 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ---------------------------------------------------#
#   SSA 分解
# ---------------------------------------------------#
def decompose_ssa(data, window_size):
    """使用 TruncatedSVD 对时序数据进行奇异谱分析分解"""
    lagged_data = pd.concat([data.shift(i) for i in range(window_size)], axis=1).dropna()
    svd = TruncatedSVD(n_components=window_size - 1)
    svd.fit(lagged_data.values)
    components = svd.transform(lagged_data.values)
    component_names = ['component_%d' % i for i in range(1, window_size)]
    components_df = pd.DataFrame(components, index=data.index[window_size - 1:], columns=component_names)
    return components_df


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


# ---------------------------------------------------#
#   PyTorch 模型
# ---------------------------------------------------#
class SSALSTMModel(nn.Module):
    """SSA 分解后的 LSTM 单步预测模型"""

    def __init__(self, input_dim, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        return self.fc(out[:, -1, :])  # 取最后时间步 → (batch, input_dim)


# ---------------------------------------------------#
#   主流程
# ---------------------------------------------------#
if __name__ == "__main__":
    # 加载数据
    data_path = PROJECT_ROOT / 'dataSet' / 'TE transition mode data' / 'M1M2XMEAS.mat'
    data = scipy.io.loadmat(str(data_path))
    df = pd.DataFrame(data['simout'])
    components = decompose_ssa(df, window_size=12)

    # 划分训练集/测试集
    train_size = int(len(components) * 0.8)
    train_data = components[:train_size]
    test_data = components[train_size:]

    # 标准化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # 滑动窗口
    n_steps = 12
    X_train, y_train = split_sequences(train_data, n_steps)
    X_test, y_test = split_sequences(test_data, n_steps)

    # 转为 PyTorch tensor
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)

    # 模型
    model = SSALSTMModel(input_dim=X_train.shape[2], hidden_size=50).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 训练
    model.train()
    for epoch in range(10000):
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/10000, Loss: {loss.item():.6f}")

    # 预测
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()

    # 反标准化
    y_pred = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test)

    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    print('RMSE: %.3f' % rmse)

    # 保存最优模型
    rmse_csv = RESULTS_DIR / 'rmse.csv'
    model_pt = SAVED_MODELS_DIR / 'ssa_lstm.pt'

    rmse_old = float('inf')
    if os.path.exists(rmse_csv):
        df_csv = pd.read_csv(rmse_csv)
        if len(df_csv) > 0:
            rmse_old = df_csv.iloc[-1, 1]

    if rmse_old > rmse:
        torch.save(model.state_dict(), str(model_pt))
        pd.DataFrame({'rmse': [rmse]}).to_csv(rmse_csv, index=False)
        print("模型已保存")

    # 绘制前 100 个时间步
    x = np.arange(min(100, len(y_pred) - 300))
    for i in range(min(3, y_pred.shape[1])):  # 只画前 3 个维度
        plt.figure()
        plt.plot(x, y_pred[300:300 + len(x), i], linestyle='-', label='predicted')
        plt.plot(x, y_test_orig[300:300 + len(x), i], linestyle='-', marker='o', label='actual')
        plt.legend()
        plt.title(f'Component {i + 1}')
        plt.show()
