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
import math
import os


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
        return self.data[0][idx], self.data[1][idx]

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
    # 得到不同的数据集
    dataset_train = my_dataset([train_x_set, train_y_set], 'train')
    dataset_test = my_dataset([test_x_set, test_y_set], 'test')
    dataset_valid = my_dataset([valid_x_set, valid_y_set], 'valid')

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


class TransFormer(nn.Module):

    def __init__(self, d_model=128, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512,
                 dropout=0.1, feature_num=7, output_size=96):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.output_size = output_size
        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=d_model)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.input_projection = nn.Linear(feature_num, d_model)
        self.output_projection = nn.Linear(feature_num, d_model)
        self.fc = nn.Linear(d_model, feature_num)  # 转化为最后的输出

    # encoder的处理
    def encode_src(self, src):
        # 输出维度为[in_seq_len,batch_size,d_model]
        src_start = self.input_projection(src).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )  # 维度为[batch，in_sequence_len]
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        # pos_encoder 可变的位置编码
        src = src_start + pos_encoder  # 编码之后加上位置编码
        print(src.shape)
        src = self.encoder(src) + src_start  # 残差连接
        return src

    # decoder的处理
    def decode_trg(self, trg, memory):
        # 相当于embedding编码
        trg_start = self.output_projection(trg).permute(1, 0, 2)
        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)
        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        trg = pos_decoder + trg_start  # 位置编码加上原来的embedding

        trg_mask = self.gen_trg_mask(out_sequence_len, trg.device)
        print(trg.shape)
        # 这里采用mask掩码，并采用残差连接，这个掩码用在了self attention中，防止看后面的，对于每一个元素来说
        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start
        out = out.permute(1, 0, 2)
        out = self.fc(out)
        # 这里输出相对应的结果，这个结果和decoder的输入结果维度是相同的。
        return out

    # mask处理
    def gen_trg_mask(self, length, device):
        mask = torch.tril(torch.ones(length, length, device=device)) == 1
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    # 整个模型的操作
    def forward(self, enc_x, dec_y = None):
        batch_size, features_size = enc_x.shape[0], enc_x.shape[2]
        src = self.encode_src(enc_x)
        decoder_input = enc_x[:, -1, :].reshape(batch_size, 1, features_size)

        # 训练和测试有两种不同的形式
        if dec_y is not None:
            # decoder的输入是encoder的输入最后时刻的元素加上decoder原本的元素，并去除最后一个元素
            dec_y_final = torch.cat((decoder_input, dec_y[:, :-1, :]), dim=1)
            outputs = self.decode_trg(trg=dec_y_final, memory=src)
        else:
            # 测试，一个一个预测
            outputs = torch.zeros(batch_size, self.output_size, features_size)
            for t in range(self.output_size):
                decoder_output = self.decode_trg(trg=decoder_input, memory=src)
                outputs[:, t, :] = decoder_output.reshape(batch_size, features_size)
                decoder_input = decoder_output
        return outputs




# train
def train(net, train_iter, valid_iter, epochs, learning_rate, device, scaler_model):
    train_loss_list = []
    valid_loss_list = []
    bast_loss = np.inf
    # MSE 均方差
    loss_function = nn.L1Loss()  # 定义损失函数
    # MAE 均绝对值误差
    # torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    net.to(device)
    loss_function.to(device)
    for epoch in range(epochs):
        net.train()
        train_bar = tqdm(train_iter)
        train_loss = 0  # 均方误差

        for x_train, y_train in train_bar:
            optimizer.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_train_pred = net(x_train,y_train)  # 强制教学

            y_train_pred = y_train_pred.to(device)
            loss = loss_function(y_train_pred, y_train).sum()
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            scheduler.step()
            train_bar.desc = f'train epoch[{epoch + 1}/{epochs}] loss:{loss}'
            train_loss_list.append(loss.detach().cpu().float())
            train_loss += loss
        print(f'train epoch[{epoch + 1}/{epochs}] all_loss:{train_loss}')
        # 评估阶段使用验证集valid_iter
        # if epoch % 10 == 0:
        net.eval()
        with torch.no_grad():
            valid_loss = 0  # 均方误差
            valid_bar = tqdm(valid_iter)
            for x_valid, y_valid in valid_bar:
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                y_train_pred = net(x_valid)  # 没有强制教学
                y_train_pred = y_train_pred.to(device)
                loss_single = loss_function(y_train_pred, y_valid).sum()
                valid_loss += loss_single
                valid_loss_list.append(loss_single.detach().cpu().float())
                valid_bar.desc = f'valid epoch[{epoch + 1}/{epochs}] loss:{loss_single}'
                # mae_loss += loss_mae
            print(f'valid epoch[{epoch + 1}/{epochs}] all_loss:{valid_loss}')
            if bast_loss > valid_loss:
                # 训练完毕之后保存模型
                torch.save(model, 'transformer/transformer_model_mae.pth')  # 保存模型
                bast_loss = valid_loss
        subplot_figure(model, dataset_train, scaler_model, device, 7,teaching_focre=False)
    return net, train_loss_list, valid_loss_list


# predict
def predict(net, test_iter, device):
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
        mse_loss = 0  # 均方误差
        mae_loss = 0
        mse_loss_list = []
        mae_loss_list = []
        for x_test, y_test in test_iter:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_test_pred = net(x_test)  # 没有强制教学
            y_test_pred = y_test_pred.to(device)
            loss_mse = MSEloss_function(y_test_pred, y_test).sum()
            loss_mae = L1loss_function(y_test_pred, y_test).sum()
            mse_loss_list.append(float(loss_mse.detach().cpu()))
            mae_loss_list.append(float(loss_mae.detach().cpu()))
            mse_loss += loss_mse
            mae_loss += loss_mae
        print(f'test all_mse_loss:{mse_loss},all_mae_loss:{mae_loss}')
    return mse_loss_list, mae_loss_list


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
def subplot_figure(model, data_loader, scaler_model, device, features_size, teaching_focre = False,save=False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x, y = data_loader[-2:-1]
        x_origin = scaler_model.inverse_transform(x.reshape(-1, features_size))[:-1]  # 前96个值
        y_origin = scaler_model.inverse_transform(y.reshape(-1, features_size))[:-1]  # 真实的后o个时间点
        if teaching_focre:
            y_predict = scaler_model.inverse_transform(model(x.to(device),y.to(device)).detach().cpu() .reshape(-1, features_size).numpy())[
                        :-1]
        else:
            y_predict = scaler_model.inverse_transform(model(x.to(device)).detach().reshape(-1, features_size).numpy())[:-1]
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
            plt.plot([i for i in range(len(x_origin) - 1, len(x_origin) + len(y_origin) - 1)], y_predict[:, index],
                     "#d70005", linewidth=0.9, label="prediction")
            plt.title(f"({index + 1}) {features[index]} Variable prediction", fontproperties="Times New Roman", fontsize=12)
            plt.ylabel('values', fontproperties="Times New Roman", fontsize=10)  # y轴标题
            plt.xlabel('times', fontproperties="Times New Roman", fontsize=10)  # y轴标题
            plt.yticks(fontproperties="Times New Roman", c='black')
            plt.xticks(fontproperties="Times New Roman", c='black')
            plt.legend(loc=3, prop={'family': 'Times New Roman', 'size': 8})
        plt.subplots_adjust(wspace=0.2, hspace=0.46)
        plt.suptitle("Variable prediction for a single sample", fontproperties="Times New Roman", fontsize=16, y=0.95)
        plt.show()
        if save:
            f.savefig(f'transformer/Variable_prediction_for_a_single_sample.svg', dpi=3000, bbox_inches='tight')


if __name__ == '__main__':
    #  获取数据
    root_path = 'ETT-small'
    input_size = 96  # 输入维度
    output_size = 96  # 预测维度
    timestep = 1  # 数据步长
    batch_size = 128  # 批量大小
    epochs = 100  # 轮次
    learning_rate = 0.01  # 学习率
    features_size = 7  # 数据特征维度
    d_model = 256  # embedding的维度
    nhead = 4  # 多头注意力机制的头数
    num_encoder_layers = 3  # encoder的层数
    num_decoder_layers = 3  # decoder的层数
    dim_feedforward = 512  # 前反馈神经网络规模
    dropout = 0.5
    type_train = True
    scaler_model = MinMaxScaler()
    dataset_train, dataset_test, dataset_valid = data_process(root_path, input_size, output_size, timestep,
                                                              scaler_model)
    # 将数据载入到dataloader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)
    # 获得设备
    device = get_device()  # 获得设备

    model = TransFormer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                        num_decoder_layers=num_decoder_layers,
                        dim_feedforward=dim_feedforward, dropout=dropout, feature_num=features_size,
                        output_size=output_size)

    model, train_loss_list, valid_loss_list = train(model, train_loader, valid_loader, epochs, learning_rate, device,
                                                    scaler_model)
    pd.Series(train_loss_list).to_csv('transformer/train_loss_list.csv', index=False, header=False)
    pd.Series(valid_loss_list).to_csv('transformer/valid_loss_list_featrure_conv_short.csv', index=False, header=False)
    # 模型预测
    mse_loss_list, mae_loss_list = predict(model, test_loader, device)
    pd.Series(mse_loss_list).to_csv('transformer/mse_loss_list.csv', index=False, header=False)
    pd.Series(mae_loss_list).to_csv('transformer/mae_loss_list.csv', index=False, header=False)
    # 画图
    # plot_figure(model, dataset_test, scaler_model, device, 7)
    subplot_figure(model, dataset_test, scaler_model, device, 7, save=True)


