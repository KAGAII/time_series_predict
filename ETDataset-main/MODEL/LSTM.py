import torch
import torch.nn as nn
# encoder
class Encoder(nn.Module):

    def __init__(self, features_size, hidden_size, num_layers):
        super().__init__()
        self.features_size = features_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.features_size, self.hidden_size, self.num_layers)
    def forward(self, X):
        X = X.permute(1, 0, 2)
        output, state = self.lstm(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state包含两个 一个隐状态一个记忆细胞,(hn, cn)
        # 两个的形状都为:(num_layers,batch_size,num_hiddens)
        return output, state

# decoder
class Decoder(nn.Module):
    def __init__(self, features_size, hidden_size, num_layers):
        super().__init__()
        self.features_size = features_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.features_size, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.features_size)
    def forward(self, X, state):
        # (batch_size,num_steps,features_size)
        X = X.permute(1, 0, 2)
        output, state = self.lstm(X, state)
        output = self.linear(output).permute(1, 0, 2)
        # output的形状:batch_size,num_steps,features_size
        # state的形状:num_layers,batch_size,num_hiddens
        return output, state

    # 获取encoder的输出状态
    def init_state(self, enc_outputs):
        return enc_outputs[-1]

# encoder_decoder
class Encoder_Decoder(nn.Module):
    def __init__(self, features_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.encoder = Encoder(features_size, hidden_size, num_layers)
        self.decoder = Decoder(features_size, hidden_size, num_layers)
        self.output_size = output_size  # 预测步长

    def forward(self, enc_x, dec_y=None):
        batch_size, features_size = enc_x.shape[0], enc_x.shape[2]
        enc_outputs = self.encoder(enc_x)
        dec_state = self.decoder.init_state(enc_outputs)  # 得到decoder的初始状态
        outputs = torch.zeros(batch_size, self.output_size, features_size)
        if dec_y is not None:# training
            outputs, dec_state = self.decoder(dec_y, dec_state)
        else:# validation or test
            decoder_input = torch.zeros((batch_size, 1, features_size)).to(enc_x.device)
            for t in range(self.output_size):
                decoder_output, dec_state = self.decoder(decoder_input, dec_state)
                outputs[:, t, :] = decoder_output.reshape(batch_size, features_size)
                decoder_input = decoder_output
        return outputs


