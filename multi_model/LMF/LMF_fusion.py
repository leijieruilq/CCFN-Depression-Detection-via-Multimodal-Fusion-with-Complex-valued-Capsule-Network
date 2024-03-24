import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from single_model.resnet import Resnet
from single_model.bilstm import TextBiLSTM
from LMF import lmf_model

class LMF(nn.Module):
    def __init__(self,configs_audio,configs_text,configs_fusion):
        super(LMF,self).__init__()
        self.resnet = Resnet(configs_audio)
        self.biltsm = TextBiLSTM(configs_text)
        self.lmf = lmf_model(configs_fusion)
    
    def forward(self,audio_in,text_in):
        x_audio = self.resnet(audio_in)
        x_text = self.biltsm(text_in)
        out = self.lmf(x_audio,x_text)
        return out
# def test_LMF():
#     configs_audio = {"audio_c_in":3,"audio_c_out":320}
#     configs_text = {'text_dropout':0.3,
#                 'text_rnn_layers': 2,
#                 'text_hidden_dims':320,
#                 'text_embedding_size': 768,
#                 'text_bidirectional': True}
#     configs_fusion={"audio_dim":320,
#              "text_dim":320,
#              "fusion_dim":128,
#              "hidden_dim":32,
#              "rank":4,
#              "num_classes":2}
#     audio_x = torch.randn(32,3,256)
#     text_x = torch.randn(32,3,768)
#     model = LMF(configs_audio,configs_text,configs_fusion)
#     print(model(audio_x,text_x).shape)
# test_LMF()