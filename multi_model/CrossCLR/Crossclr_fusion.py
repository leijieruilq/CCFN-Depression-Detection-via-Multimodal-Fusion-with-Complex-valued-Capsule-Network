import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from single_model.resnet import Resnet
from single_model.bilstm import TextBiLSTM
from Crossclr import CrossCLR_onlyIntraModality

class Crossclr(nn.Module):
    def __init__(self,configs_audio,configs_text,configs_fusion):
        super(Crossclr,self).__init__()
        self.resnet = Resnet(configs=configs_audio)
        self.bilstm = TextBiLSTM(config=configs_text)
        self.mlp = nn.Linear(configs_fusion["fusion_dim"],configs_fusion["num_classes"])
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
        return out,x_audio,x_text

# def test_crossclr():
#     configs_audio = {"audio_c_in":3,"audio_c_out":320}
#     configs_text = {'text_dropout':0.3,
#                 'text_rnn_layers': 2,
#                 'text_hidden_dims':320,
#                 'text_embedding_size': 768,
#                 'text_bidirectional': True}
#     configs_fusion = {"num_classes":2,
#                       "fusion_dim":640}
#     audio_x = torch.randn(32,3,256)
#     text_x = torch.randn(32,3,768)
#     model = Crossclr(configs_audio,configs_text,configs_fusion)
#     print([i.shape for i in model(audio_x,text_x)])
# test_crossclr()