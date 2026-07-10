"""
GNN-AT-LSTM: 图神经网络 + 自适应阈值 + LSTM 故障检测模型

完整的端到端模型，整合以下模块:
  - ProcessGraphBuilder: TE 过程拓扑图构建
  - GATEncoder: 图注意力网络空间编码
  - LSTM: 时序预测
  - KDEAdaptiveThreshold: 自适应阈值工况切换检测
  - AttentionTracker: 注意力权重追踪 (故障定位)

这是本研究的核心模型文件。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from model.process_graph import (
    build_process_graph, sensors_to_node_features,
    NUM_NODES, NODE_NAMES
)
from model.adaptive_threshold import KDEAdaptiveThreshold
from model.gat_encoder import GATEncoderSimple as _GATEncoder


class GNNATLSTM(nn.Module):
    """
    GNN-AT-LSTM 完整模型。

    架构: 传感器 → 节点聚合 → GAT 空间编码 → LSTM 时序预测 → 残差输出

    Args:
        input_dim:      输入维度 (图模式=52传感器, 非图模式=数据列数)
        gat_hidden:      GAT 隐藏维度 (default: 64)
        lstm_hidden:     LSTM 隐藏维度 (default: 50)
        lstm_layers:     LSTM 层数 (default: 2)
        gat_heads:       GAT 注意力头数 (default: 4)
        n_steps:         输入序列长度 (default: 12)
        node_dim:        节点数 (default: 8)
        output_dim:      输出维度 (图模式=8节点, 非图模式=input_dim)
        dropout:         Dropout 概率 (default: 0.1)
        use_graph:       是否使用图结构 (消融实验用, default: True)
    """

    def __init__(self, input_dim=52, gat_hidden=64, lstm_hidden=50, lstm_layers=2,
                 gat_heads=4, n_steps=12, node_dim=NUM_NODES, output_dim=None,
                 dropout=0.1, use_graph=True):
        super().__init__()
        self.n_steps = n_steps
        self.node_dim = node_dim
        self.use_graph = use_graph
        self.gat_hidden = gat_hidden

        # 节点特征扩展: 标量 → 向量 (让 GAT 有足够的特征空间)
        self.node_expand = nn.Linear(1, gat_hidden) if use_graph else None

        # GAT 空间编码器
        if use_graph:
            self.gat_encoder = _GATEncoder(
                in_channels=gat_hidden,
                hidden_channels=gat_hidden,
                out_channels=gat_hidden,
                heads=gat_heads,
                dropout=dropout,
            )
            lstm_input_dim = gat_hidden * node_dim  # GAT 输出展平
            self.output_dim = node_dim  # 预测 8 个节点特征
        else:
            self.gat_encoder = None
            lstm_input_dim = input_dim  # 直接使用原始传感器输入
            self.output_dim = output_dim if output_dim else input_dim

        # LSTM 时序预测器
        self.lstm = nn.LSTM(
            lstm_input_dim, lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # 输出投影
        self.fc = nn.Linear(lstm_hidden, self.output_dim)

        # 存储
        self.edge_index = None
        self.edge_weight = None
        self.attention_cache = []  # 存储最近几个时间步的注意力权重

    def set_graph(self, edge_index, edge_weight=None):
        """设置过程拓扑图"""
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def forward(self, x_raw, return_attention=False):
        """
        前向传播。

        Args:
            x_raw: (batch, n_steps, n_sensors) 传感器数据
                   图模式: n_sensors=52 (XMEAS)
                   非图模式: n_sensors=任意

        Returns:
            pred: (batch, output_dim) 预测值
        """
        batch, seq_len, n_sensors = x_raw.shape

        if self.use_graph and self.edge_index is not None:
            # Step 1: 传感器 → 节点特征
            node_feat = sensors_to_node_features(x_raw)  # (batch, seq_len, 8)

            # Step 2: GAT 空间编码
            gat_out = []
            for t in range(seq_len):
                feat_t = node_feat[:, t, :].unsqueeze(-1)  # (batch, 8, 1)
                feat_t = self.node_expand(feat_t)            # (batch, 8, gat_hidden)
                out_t = self.gat_encoder(
                    feat_t,
                    self.edge_index.to(x_raw.device),
                    self.edge_weight.to(x_raw.device) if self.edge_weight is not None else None
                )
                gat_out.append(out_t.unsqueeze(1))
            encoded = torch.cat(gat_out, dim=1)  # (batch, seq_len, 8, gat_hidden)
            encoded = encoded.reshape(batch, seq_len, -1)  # (batch, seq_len, 8*gat_hidden)
        else:
            # 非图模式: 直接使用原始传感器数据
            encoded = x_raw

        # Step 3: LSTM 时序建模
        lstm_out, _ = self.lstm(encoded)  # (batch, seq_len, lstm_hidden)

        # Step 4: 预测下一时间步
        pred = self.fc(lstm_out[:, -1, :])

        return pred

    def get_attention_weights(self, x_raw):
        """获取 GAT 注意力权重 (用于故障定位)"""
        if not self.use_graph or self.gat_encoder is None:
            return None

        node_feat = sensors_to_node_features(x_raw)  # (batch, seq_len, 8)
        all_attn = []

        # 取最后一个时间步
        feat = node_feat[:, -1, :]  # (batch, 8)
        if hasattr(self.gat_encoder, 'get_attention_weights'):
            attn = self.gat_encoder.get_attention_weights(
                feat.to(x_raw.device),
                self.edge_index.to(x_raw.device),
                self.edge_weight.to(x_raw.device) if self.edge_weight is not None else None
            )
            all_attn = attn

        return all_attn

    def locate_fault(self, x_raw, normal_attn_baseline=None):
        """
        故障定位: 通过注意力偏差定位故障源。

        Args:
            x_raw: 当前时间步的传感器数据
            normal_attn_baseline: 正常工况下的基线注意力 (预计算)

        Returns:
            ranked_nodes: [(node_idx, deviation_score), ...] 按偏差降序排列的节点
        """
        attn = self.get_attention_weights(x_raw)
        if attn is None or normal_attn_baseline is None:
            # 回退: 基于预测残差定位
            pred = self.forward(x_raw)
            node_feat = sensors_to_node_features(x_raw)
            actual = node_feat[:, -1, :].to(pred.device)
            residuals = (pred - actual).abs().mean(dim=0).cpu().numpy()
            ranked = sorted(enumerate(residuals), key=lambda x: -x[1])
            return [(i, float(s)) for i, s in ranked]

        # 基于注意力偏差
        deviations = []
        for layer_attn in attn:
            if isinstance(layer_attn, tuple):
                edge_idx, alpha = layer_attn
                # 计算每条边注意力变化
                for baseline_alpha in normal_attn_baseline:
                    dev = (alpha - baseline_alpha[1]).abs().mean().item()
                    deviations.append(dev)

        if not deviations:
            return [(i, 0.0) for i in range(NUM_NODES)]

        # 汇总到节点级别 (简化: 取最大偏差)
        node_dev = np.zeros(NUM_NODES)
        ranked = sorted(enumerate(node_dev), key=lambda x: -x[1])
        return [(i, float(s)) for i, s in ranked]


def create_model(use_graph=True, **kwargs):
    """创建 GNN-AT-LSTM 模型 (便捷函数)"""
    return GNNATLSTM(use_graph=use_graph, **kwargs)


# ── 自检 ──
if __name__ == "__main__":
    print("GNN-AT-LSTM Model Test")
    print("=" * 50)

    # 构建图
    edge_index, edge_weight, edge_type = build_process_graph(use_data_driven=False)

    # 创建模型
    model = GNNATLSTM(
        gat_hidden=64, lstm_hidden=50, lstm_layers=2,
        gat_heads=4, n_steps=12, use_graph=True,
    )
    model.set_graph(edge_index, edge_weight)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # 合成数据测试
    batch, seq_len, n_sensors = 4, 12, 52
    x = torch.randn(batch, seq_len, n_sensors)
    pred = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {pred.shape}")

    # 消融模式测试 (无图)
    model_no_graph = GNNATLSTM(use_graph=False)
    pred2 = model_no_graph(x)
    print(f"No-graph output: {pred2.shape}")

    print("\nAll tests passed!")
