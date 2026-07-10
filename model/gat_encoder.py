"""
GAT (Graph Attention Network) 空间编码器

对 TE 过程拓扑图中的每个设备节点，通过多头注意力机制聚合邻居节点信息，
输出包含空间结构信息的节点嵌入向量。

基于 Veličković et al., 2018 (ICLR).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """
    多头图注意力网络空间编码器。

    对每个时间步的 8 节点特征图进行多层 GAT 编码，输出融合了邻居信息的节点嵌入。

    Args:
        in_channels:   输入特征维度 (=8，对应 8 个节点)
        hidden_channels: 隐藏层维度 (default: 64)
        out_channels:  输出嵌入维度 (default: 64)
        heads:         注意力头数 (default: 4)
        num_layers:    GAT 层数 (default: 2)
        dropout:       Dropout 概率 (default: 0.1)
    """

    def __init__(self, in_channels=8, hidden_channels=64, out_channels=64,
                 heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # 第一层: in_channels → hidden_channels (concat heads)
        self.layers.append(
            GATConv(in_channels, hidden_channels, heads=heads,
                    dropout=dropout, concat=True)
        )

        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads,
                        dropout=dropout, concat=True)
            )

        # 最后一层: hidden_channels*heads → out_channels (mean heads)
        if num_layers > 1:
            self.layers.append(
                GATConv(hidden_channels * heads, out_channels, heads=1,
                        dropout=dropout, concat=False)
            )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: (batch * num_nodes, in_channels) 或 (batch, num_nodes, in_channels)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) optional

        Returns:
            (batch, num_nodes, out_channels) 或 (num_nodes, out_channels)
        """
        batch_mode = x.dim() == 3
        batch_size = 1
        if batch_mode:
            batch_size, num_nodes, in_c = x.shape
            x = x.reshape(batch_size * num_nodes, in_c)

        # GAT 层前向传播 (PyG handles single-graph input)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)

        # 恢复 batch 维度
        if batch_mode:
            x = x.reshape(batch_size, num_nodes, self.out_channels)

        return x

    def get_attention_weights(self, x, edge_index, edge_weight=None):
        """
        获取各层的注意力权重（用于故障定位分析）。

        Returns:
            list of (num_edges,) 各层的平均注意力权重
        """
        attention_weights = []
        if x.dim() == 3:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        for layer in self.layers:
            # GATConv 的 message 方法内部计算注意力
            # 返回 (alpha, output) 元组
            result = layer(x, edge_index, edge_weight, return_attention_weights=True)
            if isinstance(result, tuple):
                x, alpha = result
                attention_weights.append(alpha)
            else:
                x = result
            x = F.elu(x)

        return attention_weights


class GATEncoderSimple(nn.Module):
    """
    简化版图编码器 — 使用两层 MLP + 邻居均值池化实现消息传递。

    当 torch_geometric 不可用时使用此版本。
    相比完整 GAT 缺少可学习的注意力权重，但保留了图结构信息聚合能力。
    """

    def __init__(self, in_channels=8, hidden_channels=64, out_channels=64,
                 heads=4, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)

        # 两层节点级 MLP (带残差)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.ELU(),
        )

        # 邻居聚合后的投影
        self.aggregate_proj = nn.Linear(out_channels * 2, out_channels)

    def _aggregate_neighbors(self, x, edge_index):
        """
        对每个节点计算邻居特征的均值聚合（向量化版本）。

        Args:
            x: (num_nodes, channels) 节点特征
            edge_index: (2, num_edges) 边索引 (src→dst)

        Returns:
            (num_nodes, channels) 聚合后的特征
        """
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # 使用 scatter_add 向量化聚合邻居特征
        aggregated = x.clone()  # 自环
        counts = torch.ones(num_nodes, 1, device=x.device)

        # scatter_add: 将源节点特征加到目标节点
        aggregated = aggregated.scatter_add(0, dst.unsqueeze(-1).expand(-1, x.shape[1]), x[src])
        counts = counts.scatter_add(0, dst.unsqueeze(-1), torch.ones(len(dst), 1, device=x.device))

        return aggregated / counts.clamp(min=1)

    def forward(self, x, edge_index, edge_weight=None):
        batch_mode = x.dim() == 3

        if batch_mode:
            batch, num_nodes, _ = x.shape
            outputs = []
            for b in range(batch):
                x_b = x[b]
                # MLP 编码
                h = self.mlp(x_b)
                # 邻居聚合
                agg = self._aggregate_neighbors(h, edge_index)
                # 融合
                out = self.aggregate_proj(torch.cat([h, agg], dim=-1))
                outputs.append(out.unsqueeze(0))
            return torch.cat(outputs, dim=0)
        else:
            h = self.mlp(x)
            agg = self._aggregate_neighbors(h, edge_index)
            return self.aggregate_proj(torch.cat([h, agg], dim=-1))


# ── 自检 ──
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.process_graph import build_process_graph, sensors_to_node_features

    # 构建图
    edge_index, edge_weight, edge_type = build_process_graph(use_data_driven=False)

    # 合成数据
    batch, seq_len = 2, 12
    xmeas = torch.randn(batch * seq_len, 52)
    node_feat = sensors_to_node_features(xmeas)
    node_feat = node_feat.reshape(batch, seq_len, 8)

    print(f"Graph: {edge_index.shape[1]} edges, 8 nodes")
    print(f"Input: {node_feat.shape}")

    try:
        from torch_geometric.nn import GATConv
        encoder = GATEncoder(in_channels=8, hidden_channels=64, out_channels=64, heads=4)
        print("Using torch_geometric GATEncoder")
    except ImportError:
        encoder = GATEncoderSimple(in_channels=8, hidden_channels=64, out_channels=64, heads=4)
        print("Using simple GATEncoder (torch_geometric not available)")

    # 逐时间步编码
    outputs = []
    for t in range(seq_len):
        out = encoder(node_feat[:, t, :], edge_index, edge_weight)
        outputs.append(out.unsqueeze(1))
    final = torch.cat(outputs, dim=1)
    print(f"Output: {final.shape}")
