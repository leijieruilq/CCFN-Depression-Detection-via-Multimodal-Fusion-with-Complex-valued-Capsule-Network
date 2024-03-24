import math
import torch
import torch.nn as nn
from torch.nn import functional as F
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# 平铺
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # shared MLP 多层感知机
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))		
                # avg_pool size: [bs, gate_channels, 1, 1]
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
            	# avg_pool + max_pool
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    # [bs, channel, w, h] to [bs, channel, w*h]
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
    	# 沿着原feature的channel分别做max_pool和avg_pool，然后将两者concat
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# In[]
# 残差操作 
class Residual(nn.Module):
    # 可以设定输入通道数、输出通道数、是否使用额外的1x1卷积层来修改通道数以及卷积层的步幅。
    def __init__(self, input_channels, out_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        
        # 2个有相同输出通道数的3×3卷积层，第一个可以指定stride
        # kernel_size=3 padding=1 说明高宽不变
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)#默认stride = 1
        
        # 是否使用额外的1x1卷积层来修改通道数 （input_channels和out_channels不一样的话，就要通过conv3变成一样的）
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
            
        # 批量归一化层
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.CBAM = CBAM(out_channels)
    
    
    #__call__函数会默认调用forward函数
    def forward(self, X):
        
        Y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)),inplace=True)))
        
        '''
        加通道注意力和空间注意力
        '''
        Y = self.CBAM(Y) * Y
        
        if self.conv3:# conv3通过改变通道数，使X和Y通道数一样，这样可以相加
            X = self.conv3(X) #如果有1x1的矩阵的话 就通过conv3改写输入
        
        Y += X  #加入残差
        return F.relu(Y)  #最后做一个relu输出
# In[]
"""
ResNet使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
第一个模块的通道数同输入通道数一致。其他模块通道数均加倍
"""
#这里定义的是一个“stage” 一个stage里有多个残差块
#输入的channel数，输出的channel数，resnet_block里包含num_residuals个残差块，你是不是第一个

def resnet_block(input_channels, out_channels, num_residuals, first_block=False):
    
    block = []
    for i in range(num_residuals):# 包含num_residuals个残差块
        if i == 0 and not first_block: # 如果是第一个block(b2)那么就不用减半了
            block.append(Residual(input_channels, out_channels, use_1x1conv=True, strides=2))#高宽减半
        else:
            block.append(Residual(out_channels, out_channels))#通道数同输入通道数一致
            
    return block

class Resnet(nn.Module):
        def __init__(self,configs):
            super(Resnet,self).__init__()
            self.b1 = nn.Sequential(nn.Conv2d(configs["audio_c_in"], 64, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            # 模块1，不适用1X1卷积  *是指把python里面的list展开
            self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)) #通道数没有变 不高宽减半
            
            # 模块2，需要用1x1卷积
            self.b3 = nn.Sequential(*resnet_block(64, 128, 2))#通道加倍，高宽减半    
            
            # 模块3，需要用1x1卷积
            self.b4 = nn.Sequential(*resnet_block(128, 256, 2))#通道加倍，高宽减半
            
            # 模块4，需要用1x1卷积
            self.b5 = nn.Sequential(*resnet_block(256,configs["audio_c_out"], 2))#通道加倍，高宽减半

        # ResNet-18
            self.resnet = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5,
                                nn.AdaptiveAvgPool2d((1, 1)),#全局的平均池化 把7*7变成1*1
                                nn.Flatten())
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
        def forward(self,x):
            x=x.reshape(-1,3,16,16)
            for layer in self.resnet:
                x = layer(x)
            return x
# configs={"audio_c_in":3,"audio_c_out":320}
# model = Resnet(configs)
# print(model(torch.randn(32,3,256)).shape)