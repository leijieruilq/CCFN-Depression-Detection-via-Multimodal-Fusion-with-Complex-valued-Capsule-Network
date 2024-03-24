import torch
import numpy as np
import torch.nn as nn

class MyTransformer(torch.nn.Module):
    def __init__(self,configs):
        super(MyTransformer, self).__init__()
        self.embedding = torch.nn.Linear(configs["fusion_dim"], configs["tf_dim"])
        self.transformer1 = torch.nn.TransformerEncoderLayer(configs["tf_dim"], nhead=configs["tf_heads"])
        self.transformer2 = torch.nn.TransformerEncoderLayer(configs["tf_dim"], nhead=configs["tf_heads"])
        self.transformer3 = torch.nn.TransformerEncoderLayer(configs["tf_dim"], nhead=configs["tf_heads"])
        self.fc = torch.nn.Linear(configs["tf_dim"], configs["num_classes"])
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = nn.Dropout(configs["dropout"])
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # 调整维度顺序，使得时间步维度在第一维
        x = self.transformer1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.transformer2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.transformer3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = x.permute(1, 0, 2)  # 调整维度顺序，使得时间步维度在第二维
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出，作为整个序列的输出
        return x
# def test_mytransformer():
#     configs = {"fusion_dim":640,
#                "tf_dim":16,
#                "tf_heads":4,
#                "num_classes":2,
#                "dropout":0.3
#                }
#     x = torch.randn(32,1,640)
#     model = MyTransformer(configs)
#     print(model(x).shape)
# test_mytransformer()