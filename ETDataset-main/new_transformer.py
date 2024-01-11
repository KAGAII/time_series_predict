
import torch
from torch import nn, Tensor
import positional_encoder as pe
import torch.nn.functional as F


class TimeSeriesTransformer(nn.Module):

    def __init__(self,
                 input_size: int,
                 dec_seq_len: int,
                 batch_first: bool,
                 out_seq_len: int = 96,
                 dim_val: int = 128,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dropout_decoder: float = 0.2,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 2048,
                 dim_feedforward_decoder: int = 2048,
                 num_predicted_features: int = 7
                 ):
        super().__init__()

        self.dec_seq_len = dec_seq_len
        self.out_seq_len = out_seq_len
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val)

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features)

        self.positional_encoding_layer = pe.PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None)

    def forward(self, src: Tensor, tgt: Tensor= None, src_mask: Tensor = None,
                tgt_mask: Tensor = None) -> Tensor:
        batch_size, features_size = src.shape[1], src.shape[2]
        decoder_input = src[-1, :, :].reshape(1,batch_size, features_size)
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)

        # 训练和测试有两种不同的形式
        if tgt is not None:
            # decoder的输入是encoder的输入最后时刻的元素加上decoder原本的元素，并去除最后一个元素
            dec_y_final = torch.cat((decoder_input, tgt[:-1,:, :]), dim=0).to(tgt.device)
            decoder_output = self.decoder_input_layer(dec_y_final).to(tgt.device)
            decoder_output = self.decoder(tgt=decoder_output,
                                          memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
            decoder_outputs = self.linear_mapping(decoder_output)
        else:
            # 测试，一个一个预测
            decoder_outputs = torch.zeros(self.out_seq_len,batch_size,features_size)
            for t in range(self.out_seq_len):
                decoder_output = self.decoder_input_layer(decoder_input)
                decoder_output = self.decoder(tgt=decoder_output,
                                              memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
                decoder_output = self.linear_mapping(decoder_output)
                decoder_outputs[t,:, :] = decoder_output.reshape(batch_size,features_size)
                decoder_input = decoder_output
        return decoder_outputs







if __name__ == '__main__':

    output_size = 96  # 预测维度
    features_size = 7  # 数据特征维度
    X = torch.zeros((128,96, 7)).permute(1, 0, 2)
    model = TimeSeriesTransformer(input_size = features_size,dec_seq_len=output_size,batch_first=False,out_seq_len= output_size)
    Y = model(X)
    print(Y.shape)








