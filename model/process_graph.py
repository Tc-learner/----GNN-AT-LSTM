"""
TE 过程拓扑图构建模块

将 TE 过程的 52 个测量变量 (XMEAS) 和 12 个操纵变量 (XMV) 映射为
8 个设备节点，基于物理连接关系和互信息构建有向过程拓扑图。

节点映射（基于 Downs & Vogel 1993, Ricker 1996）:
  - Reactor    (A): XMEAS 1-13  — 反应器
  - Condenser  (B): XMEAS 14-22 — 冷凝器
  - Separator  (C): XMEAS 23-30 — 气液分离器
  - Compressor (D): XMEAS 31-36 — 压缩机
  - Stripper   (E): XMEAS 37-42 — 汽提塔
  - MixZone    (F): XMEAS 43-46 — 混合区
  - Feed       (G): XMEAS 47-52 — 反应器进料
  - Global     (H): XMV 1-12 + 全局连接 — 操纵变量 + 全局耦合

边类型:
  - 物理边: 基于 TE 流程的物料/能量流方向
  - 数据驱动边: 基于传感器互信息 (Mutual Information) 的 k-NN 图
"""
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix


# ── TE 过程传感器 → 设备节点映射 (41 XMEAS) ──
# 基于 Downs & Vogel (1993) TE 过程变量说明 和 Ricker (1996) 解耦控制
SENSOR_TO_NODE = {
    # Reactor (node 0): 进料流量 + 反应器状态
    # XMEAS 1-7: 进料(流1-4)、循环流(8)、反应器进料(6)、反应器压力
    **{i: 0 for i in range(0, 7)},
    7: 0,   # XMEAS 8: 反应器液位
    8: 0,   # XMEAS 9: 反应器温度
    20: 0,  # XMEAS 21: 反应器冷却水出口温度

    # Condenser (node 1): 冷凝器
    9: 1,   # XMEAS 10: 放空率
    21: 1,  # XMEAS 22: 冷凝器冷却水出口温度

    # Separator (node 2): 气液分离器
    **{i: 2 for i in range(10, 14)},  # XMEAS 11-14

    # Stripper (node 3): 汽提塔
    **{i: 3 for i in range(14, 19)},  # XMEAS 15-19

    # Compressor (node 4): 压缩机
    19: 4,  # XMEAS 20: 压缩机功率

    # Feed/Mix (node 5): 进料混合区 + 反应物分析
    **{i: 5 for i in range(0, 5)},   # XMEAS 1-5
    6: 5,  # XMEAS 7 (shared)

    # Analysis (node 6): 成分分析
    **{i: 6 for i in range(22, 41)},  # XMEAS 23-41 (成分 A-H 在流6,9,11)
}

NODE_NAMES = [
    'Reactor', 'Condenser', 'Separator', 'Stripper',
    'Compressor', 'Feed/Mix', 'Analysis', 'Global'
]

NUM_NODES = 8
NUM_SENSORS = 41     # XMEAS (标准 TE 测量变量)
NUM_ACTUATORS = 12    # XMV (操作变量, 存储在独立 .mat 文件中)


def build_physical_edges():
    """
    构建基于 TE 过程物理拓扑的有向边列表。

    物理流向 (8节点, 基于 41 XMEAS + 12 XMV):
      进料(5) ↔ 反应器(0)
      反应器(0) → 冷凝器(1)
      冷凝器(1) → 分离器(2)
      分离器(2) → 汽提塔(3) [液相]
      分离器(2) → 压缩机(4) [气相]
      压缩机(4) → 进料区(5) [回收]
      汽提塔(3) → 进料区(5) [循环]
      分析节点(6) ↔ 反应器/分离器/汽提塔 [组分耦合]
      全局节点(7) ↔ 所有节点 [操作变量影响]

    Returns:
        edge_index: (2, num_edges) 源→目标索引
        edge_weight: (num_edges,) 初始权重
    """
    edges = [
        (5, 0), (0, 5),   # Feed ↔ Reactor
        (0, 1), (1, 0),   # Reactor ↔ Condenser
        (1, 2),            # Condenser → Separator
        (2, 3),            # Separator → Stripper (液相)
        (2, 4),            # Separator → Compressor (气相)
        (4, 5),            # Compressor → Feed (回收)
        (3, 5),            # Stripper → Feed (循环)
        # 组分分析耦合 (analysis node connects to process units)
        (6, 0), (0, 6),   # Analysis ↔ Reactor
        (6, 2), (2, 6),   # Analysis ↔ Separator
        (6, 3), (3, 6),   # Analysis ↔ Stripper
        # 全局节点双向连接 (操作变量影响)
        (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6),
        (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
        # 自环
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
    ]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.ones(edge_index.shape[1])
    return edge_index, edge_weight


def build_data_driven_edges(data, n_neighbors=3, threshold=0.3):
    """
    基于传感器互信息和 k-NN 构建数据驱动的补充边。

    Args:
        data: (n_timesteps, 52) XMEAS 数据
        n_neighbors: k-NN 的邻居数
        threshold: 互信息阈值，低于此值不建边

    Returns:
        edge_index: (2, num_edges)
        edge_weight: (num_edges,) 归一化互信息值
    """
    # 计算每个传感器与其他所有传感器之间的互信息
    n_sensors = data.shape[1]
    mi_matrix = np.zeros((NUM_NODES, NUM_NODES))

    # 将传感器数据聚合到设备节点 (取均值)
    node_data = np.zeros((data.shape[0], NUM_NODES))
    for s in range(n_sensors):
        node = SENSOR_TO_NODE.get(s, 7)  # default to Global
        node_data[:, node] += data[:, s]
    # 归一化
    for n in range(NUM_NODES):
        mask = np.array([SENSOR_TO_NODE.get(s, 7) == n for s in range(n_sensors)])
        count = mask.sum()
        if count > 0:
            node_data[:, n] /= count

    # 计算节点间互信息
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            mi = mutual_info_regression(
                node_data[:, i].reshape(-1, 1),
                node_data[:, j],
                random_state=42
            )
            mi = float(mi[0]) if hasattr(mi, '__len__') else float(mi)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    # 阈值过滤 + k-NN: 每个节点保留 top-k 邻居
    edges = []
    weights = []
    for i in range(NUM_NODES):
        # 找互信息最大的 k 个邻居
        neighbors = np.argsort(mi_matrix[i])[::-1]
        count = 0
        for j in neighbors:
            if j == i:
                continue
            if mi_matrix[i, j] >= threshold and count < n_neighbors:
                edges.append((i, j))
                weights.append(mi_matrix[i, j])
                count += 1

    if len(edges) == 0:
        return torch.tensor([[], []], dtype=torch.long), torch.tensor([])

    # 归一化权重
    weights = np.array(weights)
    weights = weights / weights.max()

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def build_process_graph(data=None, use_data_driven=True, n_neighbors=3):
    """
    构建完整的 TE 过程拓扑图（融合物理边和数据驱动边）。

    Args:
        data: (n_timesteps, n_features) 可选，用于构建数据驱动边
        use_data_driven: 是否加入数据驱动边
        n_neighbors: 数据驱动边的 k-NN 邻居数

    Returns:
        edge_index: (2, num_edges) 合并后的边索引
        edge_weight: (num_edges,) 边权重
        edge_type: (num_edges,) 0=物理边, 1=数据驱动边
    """
    phys_idx, phys_w = build_physical_edges()
    phys_type = torch.zeros(phys_idx.shape[1], dtype=torch.long)

    if use_data_driven and data is not None:
        data_idx, data_w = build_data_driven_edges(data, n_neighbors=n_neighbors)
        if data_idx.shape[1] > 0:
            data_type = torch.ones(data_idx.shape[1], dtype=torch.long)
            edge_index = torch.cat([phys_idx, data_idx], dim=1)
            edge_weight = torch.cat([phys_w, data_w])
            edge_type = torch.cat([phys_type, data_type])
            return edge_index, edge_weight, edge_type

    return phys_idx, phys_w, phys_type


def sensors_to_node_features(xmeas, xmv=None):
    """
    将 XMEAS 传感器数据聚合为 8 维节点特征。

    Args:
        xmeas: (..., n_sensors) 测量变量 (标准 TE: n=41)
        xmv:   (..., 12) 操纵变量，可选

    Returns:
        node_features: (..., 8) 每个节点聚合后的特征
    """
    if isinstance(xmeas, torch.Tensor):
        features = torch.zeros(*xmeas.shape[:-1], NUM_NODES, device=xmeas.device)
        counts = torch.zeros(NUM_NODES, device=xmeas.device)
        for s in range(NUM_SENSORS):
            node = SENSOR_TO_NODE.get(s, 7)
            features[..., node] += xmeas[..., s]
            counts[node] += 1
        for n in range(NUM_NODES - 1):  # exclude Global
            if counts[n] > 0:
                features[..., n] /= counts[n]

        # Global node: XMV 均值
        if xmv is not None:
            features[..., 7] = xmv.mean(dim=-1)

        return features
    else:
        # numpy
        features = np.zeros((*xmeas.shape[:-1], NUM_NODES), dtype=np.float32)
        counts = np.zeros(NUM_NODES, dtype=np.float32)
        for s in range(NUM_SENSORS):
            node = SENSOR_TO_NODE.get(s, 7)
            features[..., node] += xmeas[..., s]
            counts[node] += 1
        for n in range(NUM_NODES - 1):
            if counts[n] > 0:
                features[..., n] /= counts[n]

        if xmv is not None:
            features[..., 7] = xmv.mean(axis=-1)

        return features


# ── 自检 ──
if __name__ == "__main__":
    print("TE Process Graph Builder")
    print(f"  Nodes: {NUM_NODES} ({', '.join(NODE_NAMES)})")
    print(f"  Sensors: {NUM_SENSORS} (XMEAS) + {NUM_ACTUATORS} (XMV)")

    phys_idx, phys_w, phys_type = build_process_graph(use_data_driven=False)
    print(f"\n  Physical edges: {phys_idx.shape[1]}")

    # 测试传感器→节点映射
    dummy_xmeas = torch.randn(5, 52)
    node_feat = sensors_to_node_features(dummy_xmeas)
    print(f"  Sensor shape: {dummy_xmeas.shape} -> Node feature shape: {node_feat.shape}")

    # 测试数据驱动边
    import scipy.io
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    mat_path = Path(__file__).resolve().parent.parent / 'normalizeResult.mat'
    if mat_path.exists():
        data = scipy.io.loadmat(str(mat_path))
        results = np.array(data['ssa_results'])
        n_src = results['n_src'][0][0][:1000, :52]
        full_idx, full_w, full_type = build_process_graph(data=n_src, use_data_driven=True)
        n_phys = (full_type == 0).sum().item()
        n_data = (full_type == 1).sum().item()
        print(f"  Combined edges: {full_idx.shape[1]} (physical: {n_phys}, data-driven: {n_data})")
    else:
        print("  (normalizeResult.mat not found, skipping data-driven edge test)")
