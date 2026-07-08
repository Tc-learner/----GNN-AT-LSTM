import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.io import loadmat
# from torchsummary import summary  # 已注释，仅用于可选的可视化


class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionModel, self).__init__()
        self.input_size = input_size

        # QKV 线性层
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, input_size)

        self.softmax = nn.Softmax(dim=1)

        self.lstm = nn.LSTM(hidden_size, int(hidden_size/2), batch_first=True, bidirectional=True, dropout=0)

    def forward(self, input_sequence):
        # 计算 Q、K、V
        query = self.query_layer(input_sequence)
        key = self.key_layer(input_sequence)
        value = self.value_layer(input_sequence)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(1, 2))
        attention_weights = self.softmax(attention_scores)

        # 加权和
        weighted_sequence = torch.matmul(attention_weights, value)
        lstm_output, hidden = self.lstm(weighted_sequence)
        # 输出线性层
        output = self.output_layer(lstm_output)

        return output, attention_weights

def data_loader(path):
    data = loadmat(path)
    print(data.keys())  # 加载mat文件中的变量信息
    results = np.array(data['ssa_results'])
    # pn =np.squeeze(data['ssa_results'])
    n_src = results['n_src'][0][0]
    return n_src

# input_size = 1  # 一个维度的时序信号
sequence_length = 1
model = AttentionModel(sequence_length,64)

# 生成输入数据（随机示例）
input_sequence = data_loader(str(Path(__file__).resolve().parent.parent / 'normalizeResult.mat'))
input_sequence = input_sequence[3000:3050,3].astype(np.float32)
# print(input_sequence.shape)

# input_sequence_tensor = torch.tensor(input_sequence)

# input_sequence = torch.from_numpy(input_sequence)
# tensor = torch.unsqueeze(torch.tensor(input_sequence, dtype=torch.float32), dim=0).unsqueeze(dim=-1)
input_sequence_tensor = torch.reshape(torch.tensor(input_sequence, dtype=torch.float32), (1, 50, 1))
# input_sequence = torch.randn(1, sequence_length, input_size)

# 前向传播
weighted_sequence, attention_weights = model(input_sequence_tensor)
# summary(model, (50,1),device="cpu")

print("输入序列形状:", input_sequence_tensor.shape)
print("加权序列形状:", weighted_sequence.shape)
print("注意力权重形状:", attention_weights.shape)
# print(weighted_sequence)


input_sequence_np = input_sequence_tensor.squeeze().detach().numpy()
weighted_sequence_np = weighted_sequence.squeeze().detach().numpy()
# print(weighted_sequence_np)
# inputsoftmax = nn.Softmax(dim=1)
# input_sequence_np = inputsoftmax(input_sequence_np)
# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(input_sequence_np, label='Input Sequence', marker='o')
plt.plot(weighted_sequence_np[:49], label='Weighted Sequence', linestyle='--', marker='x')
plt.title('Input and Weighted Output Sequences')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
