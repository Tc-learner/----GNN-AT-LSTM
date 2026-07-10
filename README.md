# GNN-AT-LSTM：基于图神经网络与自适应阈值的TE过程故障检测

## 项目简介

针对 Tennessee Eastman（TE）化工过程在多工况条件下固定阈值（τ=1.5）误报率高的问题，提出 **GNN-AT-LSTM** 方法——融合**图神经网络（GNN）空间编码**与**核密度估计（KDE）自适应阈值**的故障检测框架。

### 核心创新

| 创新点 | 方法 | 效果 |
|--------|------|------|
| 过程拓扑图 | 41传感器→8设备节点 + 37条物理边 | 显式编码设备耦合关系 |
| GAT空间编码 | 2层多头图注意力网络 | RMSE降低 18.7% |
| KDE自适应阈值 | 核密度估计替代固定 τ=1.5 | 误报率降低 94%（10.40%→0.62%） |
| 节点级多数投票 | 8节点独立KDE + ≥5/8报警 | GNN误报率 15.88%→3.64% |
| 故障检测 | 在线残差分析 + 连续确认 | 11/11检出，平均延迟5.7步 |

---

## 项目结构

```
本科生-SSA-LSTM/
├── README.md                         # 本文件
├── requirements.txt                  # Python 依赖
├── .gitignore
│
├── model/                            # 核心模型（4个模块）
│   ├── process_graph.py             #   TE过程拓扑图构建
│   ├── gat_encoder.py               #   图注意力网络编码器
│   ├── gnn_at_lstm.py               #   GNN-AT-LSTM 完整模型
│   └── adaptive_threshold.py        #   KDE自适应阈值 + 节点级联合判断
│
├── pipelines/                        # 实验管线
│   └── gnn_at_lstm_pipeline.py      #   完整实验（对比/消融/训练）
│
├── results/                          # 实验结果（7个CSV）
├── saved_models/                     # 模型保存目录
├── dataSet/                          # TE过程数据（.mat格式）
└── venv/                             # Python虚拟环境
```

---

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.11+ |
| PyTorch | 2.12+ (CUDA版本) |
| CUDA | 12.4+ |
| GPU | NVIDIA GPU (推荐 RTX 4060+ 8GB) |

## 安装

```bash
# 1. 创建虚拟环境
python3.11 -m venv venv
source venv/Scripts/activate      # Windows
# source venv/bin/activate        # Linux/Mac

# 2. 安装 PyTorch CUDA 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 3. 安装其余依赖
pip install -r requirements.txt

# 4. 验证 GPU
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

---

## 快速验证

```bash
# 逐个测试各模块（秒级完成）
python model/process_graph.py           # TE拓扑图: 8节点, 37边
python model/gat_encoder.py             # GAT编码: (4,8,64)→(4,8,64)
python model/gnn_at_lstm.py             # 完整模型: (4,12,41)→(4,8)
python model/adaptive_threshold.py      # KDE阈值 + 节点多数投票
```

---

## 运行实验

```bash
# 2×2 对比实验（GNN vs no-GNN, KDE vs fixed-threshold）
python pipelines/gnn_at_lstm_pipeline.py --mode compare

# 消融实验
python pipelines/gnn_at_lstm_pipeline.py --mode ablate

# 完整训练（可指定epochs）
python pipelines/gnn_at_lstm_pipeline.py --mode train --epochs 100

# 单模块消融（禁用GNN或KDE）
python pipelines/gnn_at_lstm_pipeline.py --mode train --no-graph   # 仅LSTM+KDE
python pipelines/gnn_at_lstm_pipeline.py --mode train --no-kde     # GNN+固定阈值
```

---

## 模型架构

```
输入: (batch, 12, 41)  XMEAS传感器数据
         │
    ┌────▼────┐  41传感器按设备映射→8节点均值
    │ 传感器聚合 │
    └────┬────┘  (batch, 12, 8)
    ┌────▼────┐  Linear(1,64): 标量→向量
    │ 节点扩展  │
    └────┬────┘  (batch, 12, 8, 64)
    ┌────▼────┐  2层GAT, 4头注意力, 37边消息传递
    │ GAT编码  │
    └────┬────┘  (batch, 12, 8×64=512)
    ┌────▼────┐  2层LSTM, hidden=50
    │ LSTM预测 │
    └────┬────┘  (batch, 8) → 预测下一时刻节点特征
    ┌────▼────┐  8节点独立KDE + ≥5/8投票
    │ KDE投票  │
    └─────────┘  报警/正常
```

---

## 实验结果

### 工况过渡检测（M1M2数据）

| 方法 | RMSE | FAR(%) | 阈值 |
|------|------|--------|------|
| LSTM（固定 τ=1.5） | 0.7847 | 10.40 | 1.50 |
| **LSTM + KDE** | 0.7796 | **0.62** | 2.15 |
| GNN-LSTM（固定 τ） | **0.6377** | 7.77 | 1.50 |
| **GNN + 节点KDE** | 0.6370 | **3.64** | 各节点独立 |

### 置信水平敏感度（3数据集）

| 数据集 | 推荐 α | RMSE | FAR(%) | KDE τ |
|--------|--------|------|--------|-------|
| M1M2 | 0.01 (99%) | 0.776 | 0.62 | 2.15 |
| M2M4 | 0.01 (99%) | 0.840 | 0.00 | 2.89 |
| M4M1 | 0.001 (99.9%) | 0.842 | 0.22 | 1.89 |

### 故障检测延迟（vs PCA）

| 方法 | 检出率 | 平均延迟 | 多工况适应性 |
|------|--------|----------|-------------|
| PCA-T² | 12/12 | 3.1步 | ❌ 工况切换时失效 |
| PCA-Q | 12/12 | **1.8步** | ❌ 工况切换时失效 |
| **LSTM+KDE** | 11/11 | 5.7步 | **✅ FAR=0.62%** |

---

## KDE策略对比

| 策略 | 报警条件 | FAR(%) | 适用场景 |
|------|----------|--------|----------|
| 全局KDE | RMSE > τ | 15.88 (GNN) | ❌ GNN下过敏感 |
| any (≥1/8) | 任意节点报警 | 88.48 | ❌ 过于敏感 |
| **majority (≥5/8)** | 多数节点报警 | **3.64** | ✅ **推荐** |
| max (8/8) | 全部节点报警 | 0.00 | ⚠️ 可能漏报 |

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | PyTorch 2.12 + CUDA 13.2 |
| GNN | 自实现 SimpleGAT（无需PyG） |
| KDE | SciPy `gaussian_kde` |
| 数据处理 | NumPy, SciPy, Pandas, Scikit-learn |
| 可视化 | Matplotlib |
| 数据格式 | MATLAB `.mat` |

---

## 论文

> **田昌喆**. 基于图神经网络与自适应阈值的TE过程多工况故障检测方法研究. 武汉纺织大学.

文件：`GNN-AT-LSTM论文_田昌喆.docx`

本工作基于以下前期研究：

> [1] **童一凡**. 基于深度学习残差的复杂工业过程故障诊断方法研究[D]. 武汉纺织大学, 2025.

---

## 联系

- 作者：田昌喆
- 前期工作：童一凡 (2215063022)，导师：王兆静 老师
- 单位：武汉纺织大学 计算机与人工智能学院
