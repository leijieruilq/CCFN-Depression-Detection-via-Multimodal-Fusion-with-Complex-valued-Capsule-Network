import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class lmf_model(nn.Module):
    def __init__(self,configs):
        super(lmf_model, self).__init__()
        self.configs = configs
        self.Wa=torch.nn.init.xavier_normal_(nn.Parameter(torch.Tensor(configs["rank"],configs["text_dim"]+1,configs["fusion_dim"])),gain=1.0)
        self.Wb=torch.nn.init.xavier_normal_(nn.Parameter(torch.Tensor(configs["rank"],configs["audio_dim"]+1,configs["fusion_dim"])),gain=1.0)
        self.Wf=torch.nn.init.xavier_normal_(nn.Parameter(torch.Tensor(1, configs["rank"])),gain=1.0)
        self.bias=torch.nn.init.xavier_normal_(nn.Parameter(torch.Tensor(1,configs["fusion_dim"])), gain=1.0)
        self.linear1=nn.Linear(configs["fusion_dim"],configs["hidden_dim"])
        self.linear2=nn.Linear(configs["hidden_dim"],configs["num_classes"])
        #self.dropout = nn.Dropout(p=0.3)  # dropout训练
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
        
    def forward(self,audio,text):
        text=torch.cat([text,torch.ones(text.shape[0],1).to(self.configs["device"])],dim=1).to(self.configs["device"])
        audio=torch.cat([audio,torch.ones(audio.shape[0],1).to(self.configs["device"])],dim=1).to(self.configs["device"])
        fusion_A=torch.matmul(text,self.Wa.to(self.configs["device"]))
        fusion_B=torch.matmul(audio,self.Wb.to(self.configs["device"]))
        fusion_AB=fusion_A*fusion_B
        x_fuse = torch.matmul(self.Wf.to(self.configs["device"]), fusion_AB.permute(1,0,2)).squeeze() + self.bias.to(self.configs["device"])
        x=self.linear2(F.relu(self.linear1(x_fuse.to(self.configs["device"]))))
        #x=self.dropout(x)
        return x     
# def test_lmf():
#     configs={"audio_dim":320,
#              "text_dim":320,
#              "fusion_dim":128,
#              "hidden_dim":32,
#              "rank":4,
#              "num_classes":2,
#              "device":"cuda:4"}
#     model = lmf_model(configs)
#     print(model(torch.randn(32,320),torch.randn(32,320)).shape)
# test_lmf()