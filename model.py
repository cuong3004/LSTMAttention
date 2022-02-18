import torch 
import torch.nn as nn

# model
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.fc = nn.Linear(embed_dim, 1)
        

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.embed_dim).permute(2, # qkv
                                                                   0, # batch
                                                                   1, # channel
                                                                   3) # embed_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale # <q,k> / sqrt(d)
        attn.softmax(dim=-1) # Softmax over embedding dim
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, self.embed_dim)
        x = self.fc(x)
        return torch.squeeze(x, -1)
    
# class Decoder(nn.Module):
#     def __init__(self, n_feature, out_feature, n_hidden, n_layers):
#         super().__init__()
#         self.n_feature = n_feature
#         self.rnn = nn.LSTM(n_feature, n_hidden, n_layers, batch_first=True)
#         self.fc = nn.Linear(int(n_hidden), out_feature)
    
#     def forward(self, x):
#         l_out, l_hidden = self.rnn(x)
#         return self.fc(l_out)
        

# class AudioLSTM(nn.Module):

#     def __init__(self, n_feature=5, out_feature=5, n_hidden=256, n_layers=2, drop_prob=0.3):
#         super().__init__()
#         self.drop_prob = drop_prob
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_feature = n_feature
#         self.lstm = nn.LSTM(n_feature, n_hidden, n_layers, dropout=self.drop_prob, batch_first=True)
#         self.lstm_de = Decoder(n_hidden, n_feature, n_hidden, n_layers)
#         self.dropout = nn.Dropout(drop_prob)
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(int(n_hidden), int(n_hidden/2))
#         self.fc2 = nn.Linear(int(n_hidden/2), out_feature)

#     def forward(self, x, hidden):
# #         print(x.shape)
#         # x.shape (batch, seq_len, n_features)
#         l_out, l_hidden = self.lstm(x, hidden)
# #         peint
#         x_z = self.lstm_de(l_out)
# #         print(x_z.shape)
#         # out.shape (batch, seq_len, n_hidden*direction)
#         out = self.dropout(l_out)
#         # out.shape (batch, out_feature)
#         out = self.fc1(out)
#         out = self.fc2(out[:, -1, :])
# #         print(out.shape)
#         # return the final output and the hidden state
#         return out, x_z

#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
#                   weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
# #         print(hidden[0].shape)
#         return hidden
# # # model
class AudioLSTM(nn.Module):

    def __init__(self, n_feature=168, out_feature=10, n_hidden=256, n_layers=2, drop_prob=0.3, len_seg=442):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature
        # self.att = SelfAttention(self.n_hidden)
        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_hidden, int(n_hidden/2))
        self.fc2 = nn.Linear(int(n_hidden/2), out_feature)

    def forward(self, x):

        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x)
        
        # l_att = self.att(l_out)
        
        # out = torch.cat((l_out[:, -1, :], l_att), axis=1)
        out = l_out[:, -1, :]
#         print(out.shape)
        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(out)
        
        # out.shape (batch, out_feature)
        out = self.fc1(out)
        out = self.fc2(out)
#         print(out.shape)
        # return the final output and the hidden state
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         print(hidden[0].shape)
        return hidden

class AudioLSTMAttention(nn.Module):

    def __init__(self, n_feature=168, out_feature=10, n_hidden=256, n_layers=2, drop_prob=0.3, len_seg=442):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature
        self.att = SelfAttention(self.n_hidden)
        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(len_seg + n_hidden, int(n_hidden/2))
        self.fc2 = nn.Linear(int(n_hidden/2), out_feature)

    def forward(self, x):

        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x)
        
        l_att = self.att(l_out)
        
        out = torch.cat((l_out[:, -1, :], l_att), axis=1)
#         print(out.shape)
        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(out)
        
        # out.shape (batch, out_feature)
        out = self.fc1(out)
        out = self.fc2(out)
#         print(out.shape)
        # return the final output and the hidden state
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         print(hidden[0].shape)
        return hidden