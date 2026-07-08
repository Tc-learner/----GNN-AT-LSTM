"""
LSTM with Adaptive Forget Gate (PyTorch)

自定义 LSTM 变体：遗忘门融入输入与期望值的差异，增强对异常输入的鲁棒性。
在每轮训练后打印自定义层的参数用于调试。
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)

# 正常输入的期望值
expected_value = 0.0


class LSTMCellWithAdaptiveForgetGate(nn.Module):
    """带有自适应遗忘门的 LSTM 单元"""

    def __init__(self, input_dim, units):
        super().__init__()
        dim_concat = input_dim + units + input_dim  # inputs + hidden + input_diff
        self.forget_gate = nn.Linear(dim_concat, units)
        self.input_gate = nn.Linear(input_dim, units)
        self.cell_gate = nn.Linear(input_dim, units)
        self.output_gate = nn.Linear(input_dim + units, units)

    def forward(self, x, state):
        h_prev, c_prev = state
        input_diff = x - expected_value

        concat_gates = torch.cat([x, h_prev, input_diff], dim=-1)
        forget = torch.sigmoid(self.forget_gate(concat_gates))

        c = c_prev * forget + \
            torch.sigmoid(self.input_gate(x)) * torch.tanh(self.cell_gate(x))

        o = torch.sigmoid(self.output_gate(torch.cat([x, h_prev], dim=-1)))
        h = o * torch.tanh(c)
        return h, (h, c)


class CustomLSTMWithAdaptiveForgetGate(nn.Module):
    """在时间维度上循环执行自定义 LSTM 单元"""

    def __init__(self, input_dim, units):
        super().__init__()
        self.cell = LSTMCellWithAdaptiveForgetGate(input_dim, units)
        self.units = units

    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.units, device=x.device)
        c = torch.zeros(batch, self.units, device=x.device)
        outputs = []
        for t in range(seq_len):
            h, (h, c) = self.cell(x[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# ── 主实验 ──
if __name__ == "__main__":
    # 合成数据
    seq_len = 10
    normal_data = np.zeros((1, seq_len, 1), dtype=np.float32)
    normal_data[0, 5, 0] = 1.0
    data_tensor = torch.from_numpy(normal_data)

    custom_lstm = CustomLSTMWithAdaptiveForgetGate(input_dim=1, units=8)
    model = nn.Sequential(custom_lstm, nn.Linear(8, 1))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 训练 (每 10 轮打印自定义 LSTM 层的参数)
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        loss = criterion(model(data_tensor), data_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:")
            for name, param in custom_lstm.cell.named_parameters():
                print(f"  {name}: shape={param.shape}, mean={param.data.mean():.4f}")
            print()

    # 预测
    model.eval()
    with torch.no_grad():
        predicted = model(data_tensor)

    print("Normal Data:")
    print(normal_data[0, :, 0])
    print("Predicted Data:")
    print(predicted.numpy()[0, :, 0])
