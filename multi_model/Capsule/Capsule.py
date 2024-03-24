import torch
from torch import nn

def squash(x, dim=-1):#本质上为一种归一化操作,
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    scale=torch.tanh(squared_norm.sqrt())
    
    return scale * x / (squared_norm.sqrt() + 1e-8)


class PrimaryCaps1(nn.Module):#初始胶囊层
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps1, self).__init__()

        # Each conv unit stands for a single capsule.
        self.conv = nn.Conv2d(in_channels=in_channels,#256
                              out_channels=out_channels * num_conv_units,#32*8==256
                              kernel_size=kernel_size,#8
                              stride=stride)#2
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: out_capsules * (batch_size, out_channels, height, weight)
        #(batch,256,12,8)-->(batch,256,3,1)-->(batch,(3*1*32),8)-->(batch,96,8)，每一个向量维度为8，这一层具有96个胶囊神经元
        out = self.conv(x)
        # Flatten out: (batch_size, out_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)


class DigitCaps1(nn.Module):#路由胶囊层
    """Digit capsule layer."""
    #input(batch,96,8)
    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing,device):
        """
        Initialize the layer.

        Args:
            in_dim:每个胶囊载体的维数。
            in_caps:输入胶囊的数量
            out_caps:胶囊数量
            out_dim:输出胶囊向量的维度。
            num_routing:	路由算法期间的迭代次数
        """
        super(DigitCaps1, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True)#(1,2,96,16,8)


    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x) 
        u_hat = u_hat.squeeze(-1) 
        temp_u_hat = u_hat.detach()
        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to(self.device)#b size(batch,2,96,1)
        for route_iter in range(self.num_routing - 1):#迭代3次
            # (batch_size, out_caps, in_caps, 1) -> Softmax along out_caps
            c = b.softmax(dim=1)#加权累加的向量(在输出维度上) #c size(batch,2,96,1)
            s = (c * temp_u_hat).sum(dim=2) 
            v = squash(s) 
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b = uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1) #size(batch,2,96,1)
        s = (c * u_hat).sum(dim=2)  
        v = squash(s) #(batch,2,16)
        return v


class CapsNet(nn.Module):
    def __init__(self,configs):
        super(CapsNet, self).__init__()
        # Conv2d layer
        self.configs = configs
        self.conv = nn.Conv2d(configs["fusion_in"],configs["fusion_c_in"],9)#输入向量为(batch,2,20,16)-->(batch,256,12,8)
        self.relu = nn.ReLU(inplace=True)

        # Primary capsulev
        self.primary_caps = PrimaryCaps1(num_conv_units=configs["num_conv_units"],
                                        in_channels=configs["fusion_c_in"],
                                        out_channels=configs["fusion_c_out"],
                                        kernel_size=8,
                                        stride=2)#-->(batch,96,8)

        # Digit capsule
        self.digit_caps = DigitCaps1(in_dim=configs["fusion_c_out"],
                                    in_caps=96,
                                    out_caps=configs["num_classes"],#分类个数
                                    out_dim=configs["fusion_out_dim"],#输出向量长度
                                    device=configs["device"],
                                    num_routing=2)

        self.decoder = nn.Sequential(
            nn.Linear(configs["fusion_out_dim"] * configs["num_classes"], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, configs["fusion_dim"]),
            nn.Sigmoid())
        
        self.init_weight()
        
    def init_weight(net):
        for name, param in net.named_parameters():
            if 'ln' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    if len(param.shape) < 2:
                        nn.init.xavier_uniform_(param.unsqueeze(0))
                    else:
                        nn.init.xavier_uniform_(param)

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.primary_caps(out)
        out = self.digit_caps(out)
        logits = torch.norm(out, dim=-1)
        pred = torch.eye(self.configs["num_classes"]).to(self.configs["device"]).index_select(dim=0, index=torch.argmax(logits, dim=1))
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * pred.unsqueeze(self.configs["num_classes"])).contiguous().view(batch_size, -1))
        return logits, reconstruction


class CapsuleLoss(nn.Module):
    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-9
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, logits, reconstructions, device):
        # 提出了重叠抑郁类型识别任务，目标可能表现出抑郁和正常状态之间的可能性，所以使用MarginLoss，
        # 用于惩罚FN和FP的识别结果。
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        loss1 = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)
        # Reconstruction loss
        loss2 = self.reconstruction_loss_scalar * self.mse(reconstructions.contiguous().view(images.shape), images.type(torch.float32)).to(device)
        # 结合2种损失
        return loss1 + loss2

# configs = {"fusion_in":2,
#            "fusion_c_in":256,
#            "num_conv_units":32,
#            "fusion_c_out":8,
#            "fusion_out_dim":16,
#             "num_classes":2,
#             "fusion_dim":640,
#             "device":"cpu"
#            }
# model = CapsNet(configs=configs)
# print(model(torch.randn(32,2,20,16))[0].shape)