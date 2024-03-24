import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from single_model.resnet import Resnet
from tfn import TFN

class TFN_MODEL(nn.Module):
    def __init__(self,configs_audio,configs_fusion):
        super(TFN_MODEL,self).__init__()
        self.resnet = Resnet(configs_audio)
        self.tfn = TFN(configs_fusion)
    
    def forward(self,audio_in,text_in):
        x_audio = self.resnet(audio_in)
        out = self.tfn(x_audio,text_in)
        return out

# def test_tfn():
#     configs_audio = {"audio_c_in":3,
#                      "audio_c_out":320}
#     configs_fusion = {"input_dims_audio":320,
#                       "input_dims_text":768,
#                       "hidden_dims_audio":128,
#                       "hidden_dims_text":128,
#                       "text_out":128,
#                       "post_fusion_dim":128,
#                       "dropout_audio":0.2,
#                       "dropout_text":0.2,
#                       "dropout_fusion":0.2,
#                       "device":"cpu",
#                       "num_classes":2}
#     model = TFN_MODEL(configs_audio,configs_fusion)
#     print(model(torch.rand(32,3,256),torch.randn(32,3,768)).shape)
# test_tfn()
