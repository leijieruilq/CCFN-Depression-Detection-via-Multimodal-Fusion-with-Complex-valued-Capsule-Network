import torch
from torch import nn
class LSTM(nn.Module):
    def __init__(self,configs):
        super(LSTM, self).__init__()
        # 定义LSTM
        self.rnn = nn.LSTM(configs["fusion_dim"],
                           configs["hidden_dim"],
                           configs["layers"])
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为1
        self.reg = nn.Sequential(
            nn.Linear(configs["hidden_dim"],
                      configs["num_classes"])
        )

    def forward(self, x):
        x, (ht,ct) = self.rnn(x)
        seq_len, batch_size, hidden_size= x.shape
        x = x.view(-1, hidden_size)
        x = self.reg(x)
        return x

# def test_lstm():
#     configs={"fusion_dim":640,
#              "hidden_dim":128,
#              "layers":1,
#              "num_classes":2}
#     model = EFLSTM(configs)
#     print(model(torch.randn(32,1,640)).shape)
# test_lstm()