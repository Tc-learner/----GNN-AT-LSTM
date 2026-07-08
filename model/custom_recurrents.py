"""
AttentionDecoder — Bahdanau 风格注意力解码器 (PyTorch)

基于 Bahdanau et al. 2014 (arXiv:1409.0473) 的加性注意力机制，
使用 GRU 风格门控循环单元在编码器输出序列上进行逐时间步解码。

原 Keras 版本为 Keras RNN 子类，本版本使用 nn.Module + 手动时间步循环。

输入:  (batch, timesteps, input_dim)  — 编码器输出序列
输出:  (batch, timesteps, output_dim)  — 解码序列
"""
import torch
import torch.nn as nn

from .tdd import time_distributed_dense


class AttentionDecoder(nn.Module):
    """
    Bahdanau 风格注意力解码器

    Args:
        input_dim:   编码器输出维度
        units:       隐藏状态和注意力矩阵维度
        output_dim:  输出空间维度
    """

    def __init__(self, input_dim, units, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.units = units
        self.output_dim = output_dim

        # ── 注意力矩阵 ──
        self.V_a = nn.Linear(units, 1, bias=False)
        self.W_a = nn.Linear(units, units, bias=False)
        self.U_a = nn.Linear(input_dim, units, bias=True)   # bias = b_a

        # ── Reset gate ──
        self.C_r = nn.Linear(input_dim, units, bias=False)
        self.U_r = nn.Linear(units, units, bias=False)
        self.W_r = nn.Linear(output_dim, units, bias=False)
        self.b_r = nn.Parameter(torch.zeros(units))

        # ── Update gate ──
        self.C_z = nn.Linear(input_dim, units, bias=False)
        self.U_z = nn.Linear(units, units, bias=False)
        self.W_z = nn.Linear(output_dim, units, bias=False)
        self.b_z = nn.Parameter(torch.zeros(units))

        # ── Proposal ──
        self.C_p = nn.Linear(input_dim, units, bias=False)
        self.U_p = nn.Linear(units, units, bias=False)
        self.W_p = nn.Linear(output_dim, units, bias=False)
        self.b_p = nn.Parameter(torch.zeros(units))

        # ── 输出投影 ──
        self.C_o = nn.Linear(input_dim, output_dim, bias=False)
        self.U_o = nn.Linear(units, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)
        self.b_o = nn.Parameter(torch.zeros(output_dim))

        # ── 初始隐藏状态投影 ──
        self.W_s = nn.Linear(input_dim, units, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, timesteps, input_dim) 编码器输出序列
        Returns:
            (batch, timesteps, output_dim) 解码输出序列
        """
        batch, timesteps, input_dim = x.shape

        # 预计算 U_a * x_t + b_a (在所有时间步共享)
        uxpb = time_distributed_dense(x, self.U_a.weight.t(), self.U_a.bias)
        # shape: (batch, timesteps, units)

        # 初始状态
        s = torch.tanh(self.W_s(x[:, 0, :]))          # (batch, units) — 初始 s0
        y = torch.zeros(batch, self.output_dim, device=x.device)  # (batch, output_dim) — 初始 y0

        outputs = []
        for t in range(timesteps):
            # ── 注意力计算 ──
            s_repeated = s.unsqueeze(1).repeat(1, timesteps, 1)  # (batch, timesteps, units)
            w_s = self.W_a(s_repeated)                             # (batch, timesteps, units)
            e = self.V_a(torch.tanh(w_s + uxpb))                   # (batch, timesteps, 1)
            a = torch.softmax(e, dim=1)                             # attention weights
            context = torch.sum(a * x, dim=1)                       # (batch, input_dim)

            # ── GRU 风格门控 ──
            r = torch.sigmoid(self.W_r(y) + self.U_r(s) + self.C_r(context) + self.b_r)
            z = torch.sigmoid(self.W_z(y) + self.U_z(s) + self.C_z(context) + self.b_z)
            s_proposal = torch.tanh(self.W_p(y) + self.U_p(r * s) + self.C_p(context) + self.b_p)
            s = (1 - z) * s + z * s_proposal                       # 新隐藏状态

            # ── 输出 ──
            y = torch.softmax(self.W_o(y) + self.U_o(s) + self.C_o(context) + self.b_o, dim=-1)
            outputs.append(y.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (batch, timesteps, output_dim)


# ── 自检 ──
if __name__ == "__main__":
    batch, seq_len, input_dim, units, output_dim = 2, 100, 104, 32, 4
    x = torch.randn(batch, seq_len, input_dim)

    # 独立测试解码器
    decoder = AttentionDecoder(input_dim, units, output_dim)
    out = decoder(x)
    print(f"Decoder: {x.shape} -> {out.shape}")

    # 与双向 LSTM 编码器组合
    class EncoderDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(104, 64, bidirectional=True, batch_first=True)
            self.decoder = AttentionDecoder(input_dim=128, units=32, output_dim=4)

        def forward(self, x):
            enc_out, _ = self.encoder(x)
            return self.decoder(enc_out)

    model = EncoderDecoder()
    out2 = model(x)
    print(f"EncoderDecoder: {x.shape} -> {out2.shape}")
