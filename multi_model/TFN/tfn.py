# In[]
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal
# In[]
class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))
        return y_3
# def test_subnet():
#     x = torch.randn(32,640)
#     model = SubNet(in_size=640,hidden_size=128,dropout=0.2)
#     print(model(x).shape)
# test_subnet()
# In[]
class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
# def test_textsubnet():
#     x = torch.randn(32,1,640)
#     model = TextSubNet(in_size=640,hidden_size=128,out_size=256)
#     print(model(x).shape)
# test_textsubnet()
# In[]
class TFN(nn.Module):
    def __init__(self, configs):
        super(TFN, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.audio_in = configs["input_dims_audio"]
        self.text_in = configs["input_dims_text"]

        self.audio_hidden = configs["hidden_dims_audio"]
        self.text_hidden = configs["hidden_dims_text"]
        self.text_out= configs["text_out"]
        self.post_fusion_dim = configs["post_fusion_dim"]

        self.audio_prob = configs["dropout_audio"]
        self.text_prob = configs["dropout_text"]
        self.post_fusion_prob = configs["dropout_fusion"]
        self.num_classes = configs["num_classes"]
        self.device = configs["device"]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_out + 1)  * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.num_classes)
        self.init_weight()
        
    def init_weight(net):
        for name, param in net.named_parameters():
            if 'ln' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    if len(param.shape) < 2:
                        nn.init.kaiming_uniform_(param.unsqueeze(0))
                    else:
                        nn.init.kaiming_uniform_(param)

    def forward(self, audio_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(self.device),
                                        requires_grad=False), audio_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(self.device),
                                       requires_grad=False), text_h), dim=1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        output = self.post_fusion_layer_3(post_fusion_y_2)
        return output
# def test_tfn():
#     configs_fusion = {"input_dims_audio":320,
#                       "input_dims_text":768,
#                       "hidden_dims_audio":128,
#                       "hidden_dims_text":128,
#                       "text_out":128,
#                       "post_fusion_dim":128,
#                       "dropout_audio":0.2,
#                       "dropout_text":0.2,
#                       "dropout_fusion":0.2,
#                       "num_classes":2}
#     model = TFN(configs_fusion)
#     print(model(torch.randn(32,320),torch.randn(32,3,768)).shape)
# test_tfn()