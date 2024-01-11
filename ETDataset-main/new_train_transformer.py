import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AdamW
import math
import os
import utils
import inference

import torch
from torch import nn, Tensor
import positional_encoder as pe
import torch.nn.functional as F
import random


def generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt,device):
    # 上述src为encoder的输入，tgt为decoder的输入，并且是batch_first == false
    # shape:[seq,batch_size,feature_nums]
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == np.inf).transpose(0, 1)
    tgt_padding_mask = (tgt == np.inf).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask




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

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None,
                tgt_mask: Tensor = None) -> Tensor:
        # batch_size, features_size = src.shape[1], src.shape[2]
        # decoder_input = src[-1, :, :].reshape(1,batch_size, features_size)
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)
        # decoder的输入是encoder的输入最后时刻的元素加上decoder原本的元素，并去除最后一个元素
        # dec_y_final = torch.cat((decoder_input, tgt[:-1,:, :]), dim=0).to(tgt.device)
        decoder_output = self.decoder_input_layer(tgt).to(src.device)
        decoder_output = self.decoder(tgt=decoder_output,
                                      memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
        decoder_outputs = self.linear_mapping(decoder_output)
        return decoder_outputs




# dataset
class my_dataset(Dataset):

    def __init__(self, data, type):
        """
        :param data: 数据
        :param type:训练集、测试集、验证集
        """
        self.type = type
        self.data = data

    def __getitem__(self, idx):
        # 下标来调用数据
        return self.data[0][idx], self.data[1][idx],self.data[2][idx]

    def __len__(self):
        return len(self.data[0])


def concat_Data(Data, scaler_model):
    # 将data合并
    lenght = [0, len(Data[0])]
    Data_total = Data[0]
    for i in Data[1:]:
        Data_total = pd.concat([Data_total, i.copy()])
        lenght.append(len(i) + lenght[-1])
    # 合并之后进行标准化
    Data_total = scaler_model.fit_transform(np.array(Data_total))
    # 标准化之后在进行拆分
    df_all = []
    for i in range(len(lenght) - 1):
        df_all.append(Data_total[lenght[i]:lenght[i + 1] - 1])
    return df_all


# 加载数据并划分数据，每次调用保证测试集不变，防止信息泄露
def data_process(root_path, input_size, output_size, timestep, scaler_model):
    '''
    :param root_path: 根目录
    :param input_size:输入的维度，默认为96
    :param output_size:每个样本的预测维度，默认为96，后面会改成336
    :param timestep: 时间步，滑动窗口
    '''
    # 获取对应类型的数据
    data_x = []
    data_y = []
    files = os.listdir(root_path)
    files_csv = sorted([f for f in files if f.endswith('.csv')])
    df_total = []
    for file in files_csv:
        df = pd.read_csv(os.path.join(root_path, file))  # 得到每个文件数据
        df = df.drop('date', axis=1)
        df_total.append(df)
    # 得到了所有数据，开始归一化
    df_all = concat_Data(df_total, scaler_model)
    for df in df_all:
        # 得到样本以及对应的数据集
        for index in range(0, len(df) - input_size - output_size, timestep):
            data_x.append(df[index:index + input_size])
            data_y.append(df[index + input_size:index + input_size + output_size])

    # 得到样本之后划分数据集
    # 每次的调用的随机种子不同，测试集永远不变，变得是训练集和验证集的数据
    train_x_set, test_x_set = train_test_split(data_x, test_size=0.2, random_state=42)
    train_y_set, test_y_set = train_test_split(data_y, test_size=0.2, random_state=42)
    # 然后在对训练集进行划分
    seed = np.random.randint(1, 50, 1)[0]
    train_x_set, valid_x_set = train_test_split(train_x_set, test_size=0.25, random_state=42)
    train_y_set, valid_y_set = train_test_split(train_y_set, test_size=0.25, random_state=42)
    train_x_set = torch.tensor(np.array(train_x_set)).to(torch.float32)
    train_y_set = torch.tensor(np.array(train_y_set)).to(torch.float32)
    test_x_set = torch.tensor(np.array(test_x_set)).to(torch.float32)
    test_y_set = torch.tensor(np.array(test_y_set)).to(torch.float32)
    valid_x_set = torch.tensor(np.array(valid_x_set)).to(torch.float32)
    valid_y_set = torch.tensor(np.array(valid_y_set)).to(torch.float32)
    features_size = train_x_set.shape[-1]

    decoder_input = train_x_set[:, -1, :].reshape(-1, 1, features_size)
    #zero_tensor = torch.zeros(decoder_input.shape)
    train_decoder_input = torch.cat((decoder_input, train_y_set[:, :-1, :]), dim=1)

    decoder_input = test_x_set[:, -1, :].reshape(-1, 1, features_size)
    #zero_tensor = torch.zeros(decoder_input.shape)
    test_decoder_input = torch.cat((decoder_input, test_y_set[:, :-1, :]), dim=1)

    decoder_input = valid_x_set[:, -1, :].reshape(-1, 1, features_size)
    #zero_tensor = torch.zeros(decoder_input.shape)
    valid_decoder_input = torch.cat((decoder_input, valid_y_set[:, :-1, :]), dim=1)


    # 得到不同的数据集
    dataset_train = my_dataset([train_x_set,train_decoder_input, train_y_set], 'train')
    dataset_test = my_dataset([test_x_set, test_decoder_input,test_y_set], 'test')
    dataset_valid = my_dataset([valid_x_set, valid_decoder_input,valid_y_set], 'valid')

    return dataset_train, dataset_test, dataset_valid


# 模型 采用seq2seq架构 encoder和decoder都采用lstm
'''
模型说明：因为考虑到有两种情况的预测，短时预测和长时预测，仅光采用lstm模型结构，需要训练预测长度的模型，内次开销太大
参考文献：https://blog.csdn.net/Cyril_KI/article/details/124943601
因此采用seq2seq架构，encoder和decoder都是采用lstm作为核心模型
'''


# 获得gpu
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'当前设备为{device}')
    return device


# 画预测图
def plot_figure(model, data_loader, scaler_model, device, features_size):
    plt.figure(figsize=(12, 8))
    model.to(device)
    model.eval()
    x, y = data_loader[-2:-1]
    x_origin = scaler_model.inverse_transform(x.reshape(-1, features_size))[:-1]
    y_origin = scaler_model.inverse_transform(y.reshape(-1, features_size))[:-1]
    y_predict = scaler_model.inverse_transform(model(x.to(device)).detach().reshape(-1, features_size).numpy())[:-1]
    x_origin = np.vstack([x_origin, y_origin[0]])

    plt.plot([i for i in range(0, len(x_origin))], x_origin, ["b", "b", "b", "b", "b", "b", "b"])
    plt.plot([i for i in range(len(x_origin) - 1, len(x_origin) + len(y_origin) - 1)], y_origin,
             ["b", "b", "b", "b", "b", "b", "b"])
    plt.plot([i for i in range(len(x_origin) - 1, len(x_origin) + len(y_origin) - 1)], y_predict, "r")
    plt.legend()
    plt.show()


# 论文形式
def subplot_figure(model, data_loader, scaler_model, device, features_size, teaching_focre=False, save=False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x, decoder_input,y = data_loader[-2:-1]
        x_origin = scaler_model.inverse_transform(x.reshape(-1, features_size))[:-1]  # 前96个值
        y_origin = scaler_model.inverse_transform(y.reshape(-1, features_size))[:-1]  # 真实的后o个时间点
        x = x.permute(1, 0, 2).to(device)
        y = y.permute(1, 0, 2).to(device)
        decoder_input = decoder_input.permute(1, 0, 2).to(device)
        # random_float_tensor = 0.02 * torch.rand(decoder_input.shape) + 0.99
        # decoder_input = decoder_input * random_float_tensor.to(device)
        if teaching_focre:
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(x, decoder_input, device)
            y_predict = scaler_model.inverse_transform(
                model(x.to(device), decoder_input,src_mask.to(device),tgt_mask.to(device)).detach().cpu().reshape(-1, features_size).numpy())[
                        :-1]
        else:
            y_predict = inference.run_encoder_decoder_inference(
                model=model, src=x.to(device), forecast_window=y.shape[0],
                batch_size=x.shape[1], device=device)
            y_predict = y_predict.permute(1, 0, 2)
            y_predict = scaler_model.inverse_transform(y_predict.detach().cpu().reshape(-1, features_size).numpy())[
                        :-1]

        x_origin = np.vstack([x_origin, y_origin[0]])
        # 上面全部变成了ndarray，因此可以一个一个看
        features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        f = plt.figure(figsize=(12, 7))
        for index in range(len(features)):
            if index == 6:
                plt.subplot(3, 1, 3)
            else:
                plt.subplot(3, 3, index + 1)
            plt.plot([i for i in range(0, len(x_origin))], x_origin[:, index], '#21266e', linewidth=0.9,
                     label="Ground Truth")
            plt.plot([i for i in range(len(x_origin) - 1, len(x_origin) + len(y_origin) - 1)], y_origin[:, index],
                     '#21266e', linewidth=0.9)
            plt.plot([i for i in range(len(x_origin) - 1, len(x_origin) + len(y_predict) - 1)], y_predict[:, index],
                     "#d70005", linewidth=0.9, label="prediction")
            plt.title(f"({index + 1}) {features[index]} Variable prediction", fontproperties="Times New Roman",
                      fontsize=12)
            plt.ylabel('values', fontproperties="Times New Roman", fontsize=10)  # y轴标题
            plt.xlabel('times', fontproperties="Times New Roman", fontsize=10)  # y轴标题
            plt.yticks(fontproperties="Times New Roman", c='black')
            plt.xticks(fontproperties="Times New Roman", c='black')
            plt.legend(loc=3, prop={'family': 'Times New Roman', 'size': 8})
        plt.subplots_adjust(wspace=0.2, hspace=0.46)
        plt.suptitle("Variable prediction for a single sample", fontproperties="Times New Roman", fontsize=16, y=0.95)
        plt.show()
        if save:
            f.savefig(f'Variable_prediction_for_a_single_sample.svg', dpi=3000, bbox_inches='tight')


# 论文形式
def single_sample(model, x,decoder_input,y, scaler_model, device, features_size=7):
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_real = model(x.to(device), decoder_input).detach().cpu() # 没转化的结果，这个是归一化的
        y_predict = scaler_model.inverse_transform(y_real.reshape(-1, features_size).numpy()) # 转化之后的结果
        x_origin = scaler_model.inverse_transform(x.detach().cpu().reshape(-1, features_size))  # 前96个值
        y_origin = scaler_model.inverse_transform(y.detach().cpu().reshape(-1, features_size))  # 真实的后o个时间点
        temp_pid = y_real[-1,:,:]
        temp_real = y[-1,:,:]
        x_origin = np.vstack([x_origin, y_origin[0]])
        # 上面全部变成了ndarray，因此可以一个一个看
        features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        f = plt.figure(figsize=(12, 7))
        for index in range(len(features)):
            if index == 6:
                plt.subplot(3, 1, 3)
            else:
                plt.subplot(3, 3, index + 1)
            plt.plot([i for i in range(0, len(x_origin))], x_origin[:, index], '#21266e', linewidth=0.9,
                     label="Ground Truth")
            plt.plot([i for i in range(len(x_origin) - 1, len(x_origin) + len(y_origin) - 1)], y_origin[:, index],
                     '#21266e', linewidth=0.9)
            plt.plot([i for i in range(len(x_origin) - 1, len(x_origin) + len(y_predict) - 1)], y_predict[:, index],
                     "#d70005", linewidth=0.9, label="prediction")
            plt.title(f"({index + 1}) {features[index]} Variable prediction", fontproperties="Times New Roman",
                      fontsize=12)
            plt.ylabel('values', fontproperties="Times New Roman", fontsize=10)  # y轴标题
            plt.xlabel('times', fontproperties="Times New Roman", fontsize=10)  # y轴标题
            plt.yticks(fontproperties="Times New Roman", c='black')
            plt.xticks(fontproperties="Times New Roman", c='black')
            plt.legend(loc=3, prop={'family': 'Times New Roman', 'size': 8})
        plt.subplots_adjust(wspace=0.2, hspace=0.46)
        plt.suptitle("Variable prediction for a single sample", fontproperties="Times New Roman", fontsize=16, y=0.95)
        plt.show()
    return y_predict







# train
def train(net, train_iter, valid_iter, epochs, learning_rate, device, scaler_model,teaching_force):
    train_loss_list = []
    valid_loss_list = []
    bast_loss = np.inf
    # MSE 均方差
    loss_function = nn.L1Loss()  # 定义损失函数
    # MAE 均绝对值误差
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.005)
    net.to(device)
    loss_function.to(device)
    for epoch in range(epochs):
        net.train()
        train_bar = tqdm(train_iter)
        train_loss = 0  # 均方误差
        for x_train,decoder_input_train, y_train in train_bar:
            optimizer.zero_grad()
            x_train = x_train.to(device).permute(1, 0, 2)
            y_train = y_train.to(device).permute(1, 0, 2)
            decoder_input_train = decoder_input_train.to(device).permute(1, 0, 2)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(x_train, decoder_input_train, device)
            y_train_pred = net(x_train, decoder_input_train, src_mask=src_mask.to(device), tgt_mask=tgt_mask.to(device))  # 强制教学
            y_train_pred = y_train_pred.to(device)
            loss = loss_function(y_train_pred, y_train)
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            scheduler.step()
            train_bar.desc = f'train epoch[{epoch + 1}/{epochs}] loss:{loss}'
            train_loss_list.append([epoch,loss.detach().cpu().float()])
            train_loss += loss
        print(f'train epoch[{epoch + 1}/{epochs}] all_loss:{train_loss}')

        # 评估阶段使用验证集valid_iter

        net.eval()
        with torch.no_grad():
            valid_loss = 0  # 均方误差
            valid_bar = tqdm(valid_iter)
            for x_valid,decoder_input_valid, y_valid in valid_bar:
                x_valid = x_valid.to(device).permute(1, 0, 2)
                y_valid = y_valid.to(device).permute(1, 0, 2)
                decoder_input_valid = decoder_input_valid.to(device).permute(1, 0, 2)
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(x_valid, decoder_input_valid,
                                                                                     device)
                prediction = net(x_valid, decoder_input_valid, src_mask=src_mask.to(device),
                                   tgt_mask=tgt_mask.to(device))  # 强制教学
                loss_single = loss_function(prediction, y_valid).sum()
                valid_loss += loss_single
                valid_loss_list.append([epoch,loss_single.detach().cpu().float()])
                valid_bar.desc = f'valid epoch[{epoch + 1}/{epochs}] loss:{loss_single}'
                # mae_loss += loss_mae
            print(f'valid epoch[{epoch + 1}/{epochs}] all_loss:{valid_loss}')
            if bast_loss > valid_loss:
                # 训练完毕之后保存模型
                torch.save(net, 'transformer_model_mae_short.pth')  # 保存模型
                bast_loss = valid_loss
        subplot_figure(net, dataset_train, scaler_model, device, 7, teaching_focre=False)
        subplot_figure(net, dataset_train, scaler_model, device, 7, teaching_focre=True)
    return net, train_loss_list, valid_loss_list


# predict
def predict(net, test_iter,device):
    print('开始预测')
    # MSE 均方差
    MSEloss_function = nn.MSELoss()  # 定义损失函数
    # MAE 均绝对值误差
    L1loss_function = nn.L1Loss()

    net.to(device)
    MSEloss_function.to(device)
    L1loss_function.to(device)
    net.eval()
    with torch.no_grad():
        mse_loss = 0 # 均方误差
        mae_loss = 0
        mse_loss_list = []
        mae_loss_list = []
        test_bar = tqdm(test_iter)
        batch = 0
        for x_test,decoder_input, y_test in test_bar:
            batch += 1
            x_test = x_test.to(device).permute(1, 0, 2)
            y_test = y_test.to(device).permute(1, 0, 2)
            decoder_input= decoder_input.to(device).permute(1, 0, 2)
            prediction = net(x_test, decoder_input)  # 强制教学
            # prediction = inference.run_encoder_decoder_inference(
            #     model=net, src=x_test, forecast_window=y_test.shape[0],
            #     batch_size=x_test.shape[1], device=device)
            loss_mse = MSEloss_function(prediction, y_test).sum()
            loss_mae = L1loss_function(prediction, y_test).sum()
            mse_loss_list.append(float(loss_mse.detach().cpu()))
            mae_loss_list.append(float(loss_mae.detach().cpu()))
            mse_loss += loss_mse
            mae_loss += loss_mae
            test_bar.desc = f'test batch[{batch}] mse loss:{loss_mse}  mae loss:{loss_mae}'
        print(f'test all_mse_loss:{mse_loss},all_mae_loss:{mae_loss}')
    return mse_loss_list,mae_loss_list



if __name__ == '__main__':
    teaching_force= 0.4
    #  获取数据
    root_path = 'ETT-small'
    input_size = 96  # 输入维度
    output_size = 96  # 预测维度
    timestep = 1  # 数据步长
    batch_size = 256  # 批量大小
    epochs = 150  # 轮次
    learning_rate = 0.01  # 学习率
    features_size = 7  # 数据特征维度
    dim_val = 64 # embedding维度
    n_encoder_layers = 1 # encoder层数
    n_decoder_layers = 1 # decoder层数
    n_heads = 2 # 多头注意力机制
    dim_feedforward_encoder = 128 # encoder前馈神经网络规模
    dim_feedforward_decoder = 128 # decoder前馈神经网络规模
    type_train = False
    scaler_model = MinMaxScaler()
    dataset_train, dataset_test, dataset_valid = data_process(root_path, input_size, output_size, timestep,
                                                              scaler_model)
    # 将数据载入到dataloader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)
    # 获得设备
    device = get_device()  # 获得设备
    if type_train:
        model = TimeSeriesTransformer(input_size=features_size, dec_seq_len=output_size, batch_first=False,
                                      out_seq_len=output_size, dim_val=dim_val,
                                      n_encoder_layers=n_encoder_layers,n_decoder_layers=n_decoder_layers,n_heads=n_heads,
                                      dim_feedforward_encoder=dim_feedforward_encoder,dim_feedforward_decoder=dim_feedforward_decoder,
                                      num_predicted_features=features_size)

        model, train_loss_list, valid_loss_list = train(model, train_loader, valid_loader, epochs, learning_rate, device,
                                                        scaler_model,teaching_force)
        pd.DataFrame(train_loss_list).to_csv('train_loss_list_short.csv', index=False, header=False)
        pd.DataFrame(valid_loss_list).to_csv('valid_loss_list_short.csv', index=False, header=False)
    else:
        model = torch.load('output/transformer_short/transformer_model_mae_short.pth')
        # 模型预测
        mse_loss_list, mae_loss_list = predict(model, test_loader, device)
        pd.Series(mse_loss_list).to_csv('mse_loss_list_short.csv', index=False, header=False)
        pd.Series(mae_loss_list).to_csv('mae_loss_list_short.csv', index=False, header=False)
        # 画图
        subplot_figure(model, dataset_test, scaler_model, device, 7, save=False,teaching_focre=False)

        # 检查单一样本
        # x, decoder_input,y = dataset_test[-2:-1]
        # x = x.permute(1, 0, 2).to(device)
        # y = y.permute(1, 0, 2).to(device)
        # decoder_input = decoder_input.permute(1, 0, 2).to(device)
        # for i in range(1,y.shape[0]):
        #     input = decoder_input[:(i+1),:,:]
        #     y_predict_real = single_sample(model, x,input,y, scaler_model, device, 7)
        #     decoder_input[i+1,:,:] = y_predict_real


