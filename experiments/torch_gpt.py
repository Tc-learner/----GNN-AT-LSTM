import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # QKV 线性层
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=1)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

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
        # 输出线性层
        attention_out = self.output_layer(weighted_sequence)
        out, _ = self.lstm(attention_out)
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out


def data_loader(path):
    data = loadmat(path)
    print(data.keys())  # 加载mat文件中的变量信息
    results = np.array(data['ssa_results'])
    # pn =np.squeeze(data['ssa_results'])
    n_src = results['n_src'][0][0]
    return n_src


def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data.flatten(), scaler


def split_data(data, batch_size):
    # 归一化
    normalized_input, scaler = normalize_data(data)
    input_list = []
    label_list = []
    for i in range(0, len(normalized_input) - batch_size):
        y = torch.reshape(torch.tensor(normalized_input[batch_size + i], dtype=torch.float32), (1, 1))
        input_sequence_tensor = torch.reshape(torch.tensor(normalized_input[i:batch_size + i], dtype=torch.float32),
                                              (1, batch_size, 1))
        input_list.append(input_sequence_tensor)
        label_list.append(y)
    return input_list, label_list, scaler


input = data_loader(str(Path(__file__).resolve().parent.parent / 'normalizeResult.mat'))
input_sequence = input[:, 3].astype(np.float32)

input_list, label_list, scaler = split_data(input_sequence, 50)
# 定义模型参数
input_size = 1  # 输入特征的维度
hidden_size = 64  # 隐藏层的大小
num_layers = 1  # LSTM层的数量
output_size = 1  # 输出特征的维度

# 初始化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 准备输入数据（假设有一些训练数据X和对应的标签y）
# 这里的输入数据X应该是形状为(batch_size, sequence_length, input_size)的张量
# 标签y应该是形状为(batch_size, output_size)的张量
# 你需要调整输入数据的维度以适应模型的期望输入形状

# 训练模型
num_epochs = 10
init_train_list = input_list[:150]
init_y_list = label_list[:150]
for epoch in range(num_epochs):
    for input_sequence_tensor, y in zip(init_train_list, init_y_list):
        # 前向传播
        outputs = model(input_sequence_tensor)

        # 计算损失
        loss = criterion(outputs, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用训练好的模型进行预测
model.eval()
test_x_list = input_list[150:400]
test_y_list = label_list[150:400]
i =0
for test_input_tensor, y_tensor in zip(test_x_list,test_y_list):
    with torch.no_grad():
        # 准备测试数据，假设test_input是形状为(batch_size, sequence_length, input_size)的张量
        test_output = model(test_input_tensor)
        if (test_output-y_tensor) > 0.05:
            print(str(i)+' this error is too high')
            print(test_output-y_tensor)
    i += 1

# test_output包含了模型对test_input的预测结果
predicted_output = scaler.inverse_transform(test_output.numpy().reshape(-1, 1))
print(predicted_output)
