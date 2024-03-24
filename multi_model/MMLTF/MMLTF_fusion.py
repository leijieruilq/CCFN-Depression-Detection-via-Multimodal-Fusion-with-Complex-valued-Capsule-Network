import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from single_model.resnet import Resnet
from single_model.bilstm import TextBiLSTM
from MyTransformer import MyTransformer

class MMLTF(nn.Module):
    def __init__(self,configs_audio,configs_text,configs_fusion):
        super(MMLTF,self).__init__()
        self.resnet = Resnet(configs_audio)
        self.bilstm = TextBiLSTM(configs_text)
        self.transformer = MyTransformer(configs_fusion)
    def forward(self,audio_in,text_in):
        x_audio = self.resnet(audio_in)
        x_text = self.bilstm(text_in)
        x_fusion = torch.cat([x_audio,x_text],dim=1).unsqueeze(dim=1)
        out = self.transformer(x_fusion)
        return out
# def test_MMLTF():
#     configs_audio = {"audio_c_in":3,"audio_c_out":320}
#     configs_text = {'text_dropout':0.3,
#                 'text_rnn_layers': 2,
#                 'text_hidden_dims':320,
#                 'text_embedding_size': 768,
#                 'text_bidirectional': True}
#     configs_fusion = {"fusion_dim":640,
#                       "tf_dim":16,
#                       "tf_heads":4,
#                       "num_classes":2,
#                       "dropout":0.3}
#     audio_x = torch.randn(32,3,256)
#     text_x = torch.randn(32,3,768)
#     model = MMLTF(configs_audio,configs_text,configs_fusion)
#     print(model(audio_x,text_x).shape)
# test_MMLTF()

