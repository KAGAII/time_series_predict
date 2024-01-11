import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import torch
import torch.nn as nn



class Encoder(nn.Module):

    def __init__(self, features_size, hidden_size, num_layers):
        super().__init__()
        self.features_size = features_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.features_size, self.hidden_size, self.num_layers)
    def forward(self, X):
        output, state = self.lstm(X)
        return output, state




class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.model_len
        self.pred_len = configs.pred_len

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, configs.d_model, configs.embed,
                                                    dropout=configs.dropout)

        self.conv = nn.Conv1d(configs.in_channels,configs.out_channels,configs.kernel_size,padding=1)

        # Encoder-only architecture
        self.encoder = Encoder(configs.d_model, configs.d_ff, configs.e_layers)
        self.projector = nn.Linear(configs.d_ff, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,inv = True,conv = True,feature = True):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        if feature:
            enc_out = self.enc_embedding(x_enc, x_mark_enc,inv = inv)  # covariates (e.g timestamp) can be also embedded as tokens
        else:
            enc_out = self.enc_embedding(x_enc,None, inv = inv)

        if conv:
            enc_out = self.conv(enc_out)
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out)

        # B N E -> B N S -> B S N
        if inv:
            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates
        else:
            dec_out = self.projector(enc_out)[:, :, :N]  # filter the covariates

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None,inv = True,conv = True,feature = True):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,inv = inv,conv = conv,feature=feature)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]



if __name__ == "__main__":
    class configs():
        seq_len = 11
        d_model = 512
        embed = 512
        pred_len = 336
        enc_in = 7
        dec_in = 7
        c_out = 7
        n_heads = 8
        e_layers = 6
        d_layers = 2
        d_ff = 512
        moving_avg = 25
        dropout = 0.1
        in_channels = 96
        kernel_size = 3
        out_channels = 336
        activation = 'gelu'
    config = configs()
    source = torch.rand(size=(32, 96, 7))
    target_in = torch.rand(size=(32, 96, 4))
    target_out = torch.rand(size=(32, 336, 7))

    model = Model(config)

    pred = model(source, target_in,inv = False)
    print(pred.shape)

