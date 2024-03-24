import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from single_model.resnet import Resnet
from single_model.bilstm import TextBiLSTM
from Capsule import CapsNet

class CapsuleNet(nn.Module):
    def __init__(self, configs_audio,configs_text,configs_fusion):
        super(CapsuleNet,self).__init__()
        self.resnet = Resnet(configs_audio)
        self.bilstm = TextBiLSTM(configs_text)
        self.capsule = CapsNet(configs_fusion)

    def forward(self,audio_in,text_in):
        x_audio = self.resnet(audio_in)
        x_text = self.bilstm(text_in)
        x_fusion = torch.cat([x_audio.unsqueeze(dim=1),
                              x_text.unsqueeze(dim=1)],dim=1).reshape(-1,2,20,16)
        out,recons = self.capsule(x_fusion)
        return out,recons,x_fusion

