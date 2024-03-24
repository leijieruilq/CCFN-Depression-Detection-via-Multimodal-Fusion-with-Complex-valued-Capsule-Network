import torch
from torch import nn
# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self,configs):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=configs["fusion_module"],
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(64,configs["fusion_dim"])
        self.mlp2 = torch.nn.Linear(configs["fusion_dim"],configs["num_classes"])
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x
# def test_cnn():
#     configs={"fusion_module":2,
#              "fusion_dim":128,
#              "num_classes":2}
#     model = CNNnet(configs=configs)
#     print(model(torch.randn(32,2,20,16)).shape)
# test_cnn()