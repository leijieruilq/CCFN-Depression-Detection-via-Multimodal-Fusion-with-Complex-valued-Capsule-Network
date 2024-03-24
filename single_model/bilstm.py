import math
import torch
import torch.nn as nn
from torch.nn import functional as F
class TextBiLSTM(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.dropout = config['text_dropout']
        self.hidden_dims = config['text_hidden_dims']
        self.rnn_layers = config['text_rnn_layers']
        self.embedding_size = config['text_embedding_size']
        self.bidirectional = config['text_bidirectional']

        self.build_model()
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

    def build_model(self):
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.GRU(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=self.bidirectional)

        self.ln1 = nn.LayerNorm(self.embedding_size)
        self.ln2 = nn.LayerNorm(self.hidden_dims)


    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        lstm_hidden = lstm_hidden.unsqueeze(1)
        atten_w = self.attention_layer(lstm_hidden)
        m = nn.Tanh()(h)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        softmax_w = F.softmax(atten_context, dim=-1)
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        # x : [len_seq, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)
        x = self.ln1(x)
        output,final_hidden_state = self.lstm_net(x)
        output = output.permute(1, 0, 2)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        atten_out = self.ln2(atten_out)
        return atten_out
# configs={'text_dropout':0.3,
#          'text_rnn_layers': 2,
#          'text_hidden_dims':320,
#          'text_embedding_size': 768,
#          'text_bidirectional': True,
#          }
# model = TextBiLSTM(configs)
# print(model(torch.randn(32,3,768)).shape)