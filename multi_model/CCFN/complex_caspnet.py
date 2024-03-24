import torch
from torch import nn
from torch.nn.functional import relu
import torch.nn.functional as F
from torch.autograd import Variable

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype)+1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexConv2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self,input):    
        return apply_complex(self.conv_r, self.conv_i, input)

def squash(x, dim=-1):#本质上为一种归一化操作,
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    scale=torch.tanh(squared_norm.sqrt())
    return scale * x / (squared_norm.sqrt() + 1e-8)

def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)

class PrimaryCaps(nn.Module):#初始胶囊层
    """初始胶囊层."""
    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        self.conv = ComplexConv2d(in_channels=in_channels,#256
                              out_channels=out_channels * num_conv_units,#32*8==256
                              kernel_size=kernel_size,#8
                              stride=stride)#2
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)


class DigitCaps(nn.Module):#路由胶囊层
    """复数路由胶囊层."""
    #input(batch,96,8)
    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing,device):
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True).type(torch.complex64).to(device)#(1,2,96,16,8)
        nn.init.xavier_uniform_(self.W)
        #W为活动矩阵，用于在不同的角度学习初始胶囊层中提取到的特征信息，与初始胶囊相乘得到高层胶囊的预测形态。


    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x) 
        u_hat = u_hat.squeeze(-1) 
        temp_u_hat = u_hat.detach()
        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to(self.device)#b size(batch,2,96,1)

        for route_iter in range(self.num_routing-1):#迭代3次

            b=b.type(torch.float64)
            c = b.softmax(dim=1)
            s = (c * temp_u_hat).sum(dim=2) 
            v = squash(s).type(torch.complex64) 
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b = uv

        b=b.real**2+b.imag**2
        c = b.softmax(dim=1) 
        u_hat=u_hat.type(torch.float64)
        s = (c * u_hat).sum(dim=2)  
        v = squash(s) 
        return v


class Complex_CapsNet(nn.Module):
    """胶囊网络的基本实现"""
    def __init__(self,configs):
        self.configs = configs
        super(Complex_CapsNet, self).__init__()
        # 融合特征提取——卷积层
        self.conv = ComplexConv2d(configs["fusion_in"], 
                                  configs["fusion_c_in"], 
                                  9)#输入向量为(batch,2,20,16)-->(batch,256,12,8)

        # 初始复数胶囊层
        self.primary_caps = PrimaryCaps(num_conv_units=configs["num_conv_units"],
                                        in_channels=configs["fusion_c_in"],
                                        out_channels=configs["fusion_c_out"],
                                        kernel_size=8,
                                        stride=2)#-->(batch,96,8)

        # 复数路由胶囊层
        self.digit_caps = DigitCaps(in_dim=configs["fusion_c_out"],
                                    in_caps=96,
                                    out_caps=configs["num_classes"],#分类个数
                                    out_dim=configs["fusion_out_dim"],#输出向量长度
                                    device=configs["device"],
                                    num_routing=2)

        # 重构网络
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
        out = complex_relu((self.conv(x)))
        out = self.primary_caps(out)
        out = self.digit_caps(out)
        logits = torch.norm(out, dim=-1) 
        pred = torch.eye(self.configs["num_classes"]).to(self.configs["device"]).index_select(dim=0, index=torch.argmax(logits, dim=1))
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * pred.unsqueeze(self.configs["num_classes"])).contiguous().view(batch_size, -1).type(torch.float32))
        return logits, reconstruction

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num,alpha=None, gamma=2,size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets, device):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(device)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
 
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class CapsuleLoss(nn.Module):
    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-9
        self.mse = nn.MSELoss(reduction='sum')
        self.entropy = nn.CrossEntropyLoss()
        self.focal = FocalLoss(2)

    def forward(self, images, labels, logits, reconstructions,device):
        # 提出了重叠抑郁类型识别任务，目标可能表现出抑郁和正常状态之间的可能性，所以使用MarginLoss，
        # 用于惩罚FN和FP的识别结果。
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # TN
        right = (logits - self.lower).relu() ** 2  # False positive
        loss1 = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)
        # Reconstruction loss，证明胶囊网络能够保留融合特征的空间信息。
        loss2 = self.reconstruction_loss_scalar * self.mse(reconstructions.contiguous().view(images.shape), images.type(torch.float32)).to(device)
        loss3 = self.entropy(logits,labels.squeeze(dim=-1).long())
        loss4 = self.focal(logits,labels,device)
        # 结合4种损失
        return loss3, loss2 + (loss1 + loss4)/2 + loss3

# configs = {"fusion_in":2,
#            "fusion_c_in":256,
#            "num_conv_units":32,
#            "fusion_c_out":8,
#            "fusion_out_dim":16,
#             "num_classes":2,
#             "fusion_dim":640,
#             "device":"cpu"
#            }
# model = Complex_CapsNet(configs=configs)
# print(model(torch.randn(32,2,20,16).type(torch.complex64))[0].shape)
    
    
    
    
    
    
    
    
    
