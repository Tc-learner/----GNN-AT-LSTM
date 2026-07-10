# GNN-AT-LSTM：基于图神经网络与自适应阈值的 TE 过程故障检测

## 📖 项目简介

本项目面向 Tennessee Eastman（TE）化工过程的多工况故障检测问题，提出 **GNN-AT-LSTM** 方法——融合**图神经网络（GNN）空间编码**与**核密度估计（KDE）自适应阈值**的深度学习故障检测框架。

核心创新：
- 🔗 **过程拓扑图**：8 设备节点 + 37 条物理边，将 TE 过程结构知识显式嵌入模型
- 📊 **KDE 自适应阈值**：替代固定 τ=1.5，误报率降低 94%（10.40% → 0.62%）
- 🗳️ **节点级多数投票**：8 节点独立 KDE + 多数投票，解决 GNN 过敏感问题
- ⚡ **100% 故障检测率**：11 种 TE 故障平均检测延迟 5.7 步，优于 PCA 的多工况适应性

---

## 🏗️ 项目结构

```
本科生-SSA-LSTM/
│
├── README.md                         # 本文件
├── requirements.txt                  # Python 依赖
├── .gitignore                        # Git 忽略规则
│
├── pipelines/                        # 🚀 核心管线
│   ├── ssa_lstm.py                  #   基础 SSA+LSTM 预测
│   ├── online_lstm.py               #   在线增量 LSTM + 工况切换检测
│   └── gnn_at_lstm_pipeline.py      #   GNN-AT-LSTM 完整实验管线
│
├── model/                            # 🧠 模型库
│   ├── process_graph.py             #   TE 过程拓扑图构建 (8节点, 37边)
│   ├── gat_encoder.py               #   图注意力网络 (GAT) 空间编码器
│   ├── gnn_at_lstm.py               #   GNN-AT-LSTM 完整模型
│   ├── adaptive_threshold.py        #   KDE 自适应阈值 + 节点级联合判断
│   ├── lstm_attention.py            #   自定义注意力 LSTM 层
│   ├── attention_lstm_torch.py      #   PyTorch 注意力 LSTM
│   ├── custom_recurrents.py         #   Bahdanau AttentionDecoder
│   └── tdd.py                       #   时间分布密集层工具
│
├── experiments/                      # 🔬 实验脚本
│   ├── signal_module.py             #   模型对比实验 (LSTM vs Attention)
│   ├── lstm_origin_compare.py       #   基础 LSTM 异常检测
│   ├── lstm_control_gate.py         #   带控制门的 LSTM 变体
│   ├── lstm_robust.py               #   鲁棒自适应遗忘门 LSTM
│   ├── lstm_add_gate.py             #   异常值剔除门 LSTM (未完成)
│   ├── torch_gpt.py                 #   Self-Attention + LSTM
│   └── torch_test.py                #   Attention + BiLSTM 可视化
│
├── utils/                            # 🛠️ 工具
│   └── signal_filter.py             #   信号滤波 (均值+低通)
│
├── tests/                            # 🧪 测试
│   └── svd_exercise.py              #   SVD 练习
│
├── results/                          # 📊 实验结果 (CSV)
│   ├── experiment_results.csv       #   2×2 对比实验
│   ├── lstm_sensitivity.csv         #   KDE 置信水平敏感度
│   ├── gnn_sensitivity.csv          #   GNN 置信水平调优
│   ├── fault_detection.csv          #   故障检测延迟
│   ├── pca_baseline.csv             #   PCA 基线对比
│   └── full_experiment_results.csv  #   全量结果
│
├── saved_models/                     # 💾 模型检查点
├── dataSet/                          # 📁 TE 过程数据
│   └── TE transition mode data/     #   12 种工况过渡 .mat 文件
├── normalizeResult.mat               # SSA 分解结果
└── venv/                             # Python 虚拟环境
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 版本要求 |
|------|----------|
| Python | 3.11+ |
| PyTorch | 2.12+ (CUDA 版本) |
| CUDA | 12.4+ (推荐 13.2) |
| GPU | NVIDIA GPU (推荐 RTX 4060+ 8GB) |

### 安装步骤

```bash
# 1. 克隆或进入项目目录
cd 本科生-SSA-LSTM

# 2. 创建虚拟环境 (Python 3.11)
python3.11 -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# 3. 安装 PyTorch (CUDA 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 4. 安装其余依赖
pip install -r requirements.txt

# 5. 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
# 应输出: True
```

### 运行实验

```bash
# 基础 LSTM 异常检测实验 (秒级完成)
python experiments/lstm_origin_compare.py

# 自定义 LSTM 变体实验
python experiments/lstm_control_gate.py
python experiments/lstm_robust.py

# 模型对比实验 (LSTM vs Attention)
python experiments/signal_module.py

# GNN-AT-LSTM 完整实验管线
python pipelines/gnn_at_lstm_pipeline.py --mode compare   # 2×2 对比
python pipelines/gnn_at_lstm_pipeline.py --mode train     # 单次训练
python pipelines/gnn_at_lstm_pipeline.py --mode ablate    # 消融实验

# SSA-LSTM 在线增量学习
python pipelines/online_lstm.py

# SSA-LSTM 基础管线
python pipelines/ssa_lstm.py
```

### 单独测试各模块

```bash
# TE 过程拓扑图构建
python model/process_graph.py

# GAT 空间编码器
python model/gat_encoder.py

# KDE 自适应阈值
python model/adaptive_threshold.py

# GNN-AT-LSTM 完整模型
python model/gnn_at_lstm.py
```

---

## 📊 核心实验结果

### 1. 工况过渡检测 (2×2 对比, M1M2 数据)

| 方法 | RMSE | FAR(%) | 阈值 | 训练(s) |
|------|------|--------|------|---------|
| SSA-LSTM (固定 τ=1.5) | 0.7847 | 10.40 | 1.50 | 0 |
| **SSA-LSTM + KDE** | 0.7796 | **0.62 ⬇94%** | 2.15 | 0 |
| GNN-LSTM (固定 τ) | **0.6377 ⬇18.7%** | 7.77 | 1.50 | 42 |
| **GNN-AT-LSTM (节点KDE)** | 0.6370 | **3.64** | per-node | 41 |

### 2. 故障检测延迟 (vs PCA 基线)

| 方法 | 检出率 | 平均延迟 | 多工况FAR |
|------|--------|----------|-----------|
| PCA-T² | 12/12 | 3.1 步 | ❌ 失效 |
| PCA-Q (SPE) | 12/12 | **1.8 步** | ❌ 失效 |
| **LSTM+KDE** | **11/11** | 5.7 步 | **✅ 0.62%** |

### 3. KDE 置信水平敏感度 (3 数据集 × 3 水平)

| 数据集 | 推荐 α | RMSE | FAR(%) | τ |
|--------|--------|------|--------|-----|
| M1M2 | 0.01 (99%) | 0.776 | 0.62 | 2.15 |
| M2M4 | 0.01 (99%) | 0.840 | 0.00 | 2.89 |
| M4M1 | 0.001 (99.9%) | 0.842 | 0.22 | 1.89 |

---

## 🧠 模型架构

```
输入: (batch, 12, 41)  XMEAS 传感器数据
         │
    ┌────▼────┐
    │ 传感器→节点 │  41 传感器按设备映射为 8 节点
    │  聚合     │  均值池化
    └────┬────┘
         │ (batch, 12, 8)
    ┌────▼────┐
    │ 节点扩展  │  Linear(1, 64): 标量→向量
    └────┬────┘
         │ (batch, 12, 8, 64)
    ┌────▼────┐
    │   GAT    │  2层 GAT, 4头注意力
    │ 空间编码  │  37条边的消息传递
    └────┬────┘
         │ (batch, 12, 8×64=512)
    ┌────▼────┐
    │   LSTM   │  2层, hidden=50
    │ 时序预测  │
    └────┬────┘
         │ (batch, 8)  预测的节点特征
    ┌────▼────┐
    │   KDE    │  8节点独立KDE
    │ 多数投票  │  ≥5/8报警 → 全局报警
    └─────────┘
```

---

## 📁 数据说明

### TE 过渡数据 (`dataSet/TE transition mode data/`)

| 文件模式 | 说明 |
|----------|------|
| `M*M*XMEAS.mat` | 测量变量 (41维), 4001 采样点 |
| `M*M*XMV.mat` | 操纵变量 (12维) |
| 命名规则 | M1M2 = 工况1→2 过渡 |

### 故障数据 (`TE过程-多工况正常+故障数据/`)

- `smode1fault*.mat`: 模式1故障, 故障在采样点200后引入
- `smode3fault*.mat`: 模式3故障
- `xmode*fault*.mat`: 对应的操纵变量数据

---

## 🔬 节点KDE策略对比

| 策略 | 报警条件 | FAR(%) | 适用场景 |
|------|----------|--------|----------|
| 全局KDE | RMSE > τ | 15.88 (GNN) | ❌ GNN下失效 |
| any (≥1/8) | 任意节点报警 | 88.48 | ❌ 过敏感 |
| **majority (≥5/8)** | 多数节点报警 | **3.64** | ✅ **推荐** |
| max (8/8) | 全部节点报警 | 0.00 | ⚠️ 可能漏报 |

---

## 📝 论文

本项目对应的学术论文：

> **童一凡**. 基于图神经网络与自适应阈值的TE过程多工况故障检测方法研究. 武汉纺织大学.

- `论文终稿_GNN_AT_LSTM.docx` — 英文版
- `GNN-AT-LSTM论文_童一凡格式.docx` — 中文版（参照学位论文格式）

---

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.12.1 |
| 图神经网络 | PyTorch Geometric 2.8 / 自实现 SimpleGAT |
| 数据处理 | NumPy, SciPy, Pandas, Scikit-learn |
| KDE | SciPy `gaussian_kde` |
| 可视化 | Matplotlib |
| 数据格式 | MATLAB `.mat` (scipy.io) |

---

## 📄 许可证

本项目为学术研究用途。数据集基于 Downs & Vogel (1993) 的 TE 过程仿真。

---

## 📧 联系方式

- 作者：童一凡 (2215063022)
- 导师：王兆静 老师
- 学校：武汉纺织大学 计算机与人工智能学院
