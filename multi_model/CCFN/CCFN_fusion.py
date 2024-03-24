import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from single_model.resnet import Resnet
from single_model.bilstm import TextBiLSTM
from complex_caspnet import Complex_CapsNet

class CCFN(nn.Module):
    def __init__(self,configs_audio,configs_text,configs_fusion):
        super(CCFN,self).__init__()
        self.resnet = Resnet(configs=configs_audio)
        self.bilstm = TextBiLSTM(config=configs_text)
        self.complex_capsule = Complex_CapsNet(configs=configs_fusion)
    
    def forward(self,audio_in,text_in):
        x_audio = self.resnet(audio_in.reshape(-1,3,16,16))
        x_text = self.bilstm(text_in)
        x_fusion = torch.cat([x_audio.unsqueeze(dim=1),
                              x_text.unsqueeze(dim=1)],dim=1).reshape(-1,2,20,16)
        out,recons = self.complex_capsule(x_fusion.type(torch.complex64))
        return out,recons,x_fusion

# configs_audio = {"audio_c_in":3,"audio_c_out":320}
# configs_text = {'text_dropout':0.3,
#                 'text_rnn_layers': 2,
#                 'text_hidden_dims':320,
#                 'text_embedding_size': 768,
#                 'text_bidirectional': True}
# configs_fusion = {"fusion_in":2,
#                   "fusion_c_in":256,
#                   "num_conv_units":32,
#                   "fusion_c_out":8,
#                   "fusion_out_dim":16,
#                   "num_classes":2,
#                   "fusion_dim":640,
#                   "device":"cpu"}
# model = CCFN(configs_audio,configs_text,configs_fusion)
# print([model(torch.randn(32,3,256),torch.randn(32,3,768))[i].shape for i in range(2)])

