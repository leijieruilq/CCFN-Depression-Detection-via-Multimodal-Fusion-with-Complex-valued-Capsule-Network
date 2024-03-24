import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from single_model.resnet import Resnet
from single_model.bilstm import TextBiLSTM

class FAL(nn.Module):
    def __init__(self,configs_audio,configs_text,configs_fusion):
        super(FAL,self).__init__()
        self.resnet = Resnet(configs=configs_audio)
        self.bilstm = TextBiLSTM(config=configs_text)
        self.mlp = nn.Sequential(
                    nn.Linear(configs_fusion["fusion_dim"],configs_fusion["fusion_dim"]),
                    nn.BatchNorm1d(configs_fusion["fusion_dim"]),
                    nn.Linear(configs_fusion["fusion_dim"],configs_fusion["num_classes"]),
                    )
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
    
    def forward(self,audio_in,text_in):
        x_audio = self.resnet(audio_in.reshape(-1,3,16,16))
        x_text = self.bilstm(text_in)
        x_fusion = torch.cat([x_audio,x_text],dim=1)
        out = self.mlp(x_fusion)
        return out