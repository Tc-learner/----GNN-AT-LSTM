import torch
import torch.nn as nn


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # LSTM编码
        output, (hidden, cell) = self.lstm(input)

        # 计算注意力权重
        attn_weights = torch.softmax(self.attention(output), dim=1)

        # 计算注意力向量
        attn_vectors = torch.bmm(attn_weights.transpose(1, 2), output)

        # 将注意力向量和LSTM输出相加并通过线性层得到最终输出
        output = torch.relu(attn_vectors.squeeze(1))
        output = self.out(output)

        return output

