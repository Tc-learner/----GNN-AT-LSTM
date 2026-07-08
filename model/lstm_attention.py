"""
自定义注意力 LSTM 层 (PyTorch)

在每个时间步中，使用 Bahdanau 风格加性注意力对输入序列进行加权，
然后将注意力上下文向量与当前输入做逐元素乘法，再送入 LSTMCell。

注意：原 Keras 版本中 `adjusted_input = inputs * context_vector` 要求
input_dim == units。本实现遵循相同的语义。
"""
import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Bahdanau 风格加性注意力"""

    def __init__(self, input_dim, units):
        super().__init__()
        self.W1 = nn.Linear(input_dim, units, bias=False)
        self.W2 = nn.Linear(units, units, bias=False)
        self.V = nn.Linear(units, 1, bias=False)

    def forward(self, features, hidden):
        """
        Args:
            features: (batch, seq_len, input_dim) 整个输入序列
            hidden:   (batch, units) 当前 LSTM 隐藏状态
        Returns:
            context: (batch, input_dim) 注意力加权后的上下文向量
        """
        hidden_expanded = hidden.unsqueeze(1)                     # (batch, 1, units)
        score = torch.tanh(self.W1(features) + self.W2(hidden_expanded))  # (batch, seq_len, units)
        attn_weights = torch.softmax(self.V(score), dim=1)        # (batch, seq_len, 1)
        context = torch.sum(attn_weights * features, dim=1)       # (batch, input_dim)
        return context


class CustomLSTMWithAttention(nn.Module):
    """在每个时间步中嵌入注意力的 LSTM"""

    def __init__(self, input_dim, units):
        super().__init__()
        self.attention = AttentionLayer(input_dim, units)
        self.lstm_cell = nn.LSTMCell(input_dim, units)
        self.units = units
        self.input_dim = input_dim

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, units) 所有时间步的输出
        """
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.units, device=x.device)
        c = torch.zeros(batch, self.units, device=x.device)
        outputs = []
        for t in range(seq_len):
            context = self.attention(x, h)                 # (batch, input_dim)
            adjusted = x[:, t, :] * context                # 逐元素乘法 (要求 input_dim == units)
            h, c = self.lstm_cell(adjusted, (h, c))
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# ── 自检 ──
if __name__ == "__main__":
    batch, seq_len, input_dim, units = 2, 10, 8, 8
    x = torch.randn(batch, seq_len, input_dim)
    model = CustomLSTMWithAttention(input_dim, units)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
