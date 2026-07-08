"""
时间分布式密集层 — PyTorch 版本

对 3D 输入的每个时间片应用相同的线性变换（权重共享）。
原 Keras backend 版本用于 custom_recurrents.py 的 AttentionDecoder。
"""
import torch


def time_distributed_dense(x, w, b=None):
    """
    对 shape 为 (batch, timesteps, input_dim) 的输入 x，
    在每个时间片上应用 w, b 线性变换。

    Args:
        x: (batch, timesteps, input_dim) 张量
        w: (input_dim, output_dim) 权重矩阵
        b: (output_dim,) 偏置向量，可选

    Returns:
        (batch, timesteps, output_dim) 张量
    """
    batch, timesteps, input_dim = x.shape
    output_dim = w.shape[1]
    x_flat = x.reshape(-1, input_dim)           # (batch * timesteps, input_dim)
    out = torch.matmul(x_flat, w)               # (batch * timesteps, output_dim)
    if b is not None:
        out = out + b
    return out.reshape(batch, timesteps, output_dim)
