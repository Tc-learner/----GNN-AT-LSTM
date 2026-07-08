"""
基础 LSTM 异常检测实验 (PyTorch)

使用合成数据（含单点异常）训练 LSTM 自编码器，
比较原始序列与重建序列的差异。
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)


class LSTMAnomalyDetector(nn.Module):
    """LSTM 自编码器：输入序列 → LSTM → 逐时间步输出"""

    def __init__(self, input_dim=1, hidden_size=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)         # (batch, seq_len, hidden_size)
        return self.fc(out)            # (batch, seq_len, input_dim)


if __name__ == "__main__":
    # 合成数据：长 10 的序列，位置 5 插入一个异常值 1.0
    sequence_length = 10
    normal_data = np.zeros((1, sequence_length, 1), dtype=np.float32)
    normal_data[0, 5, 0] = 1.0
    data_tensor = torch.from_numpy(normal_data)

    model = LSTMAnomalyDetector(input_dim=1, hidden_size=8)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 训练
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(data_tensor)
        loss = criterion(output, data_tensor)
        loss.backward()
        optimizer.step()

    # 预测
    model.eval()
    with torch.no_grad():
        predicted = model(data_tensor)

    print("Normal Data:")
    print(normal_data[0, :, 0])
    print("Predicted Data (Original LSTM):")
    print(predicted.numpy()[0, :, 0])
