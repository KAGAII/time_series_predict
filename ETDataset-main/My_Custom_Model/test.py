import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import math
from CLASS.data_class_time_features import data_utils
from Conv_iLSTM import Model


# 获得gpu
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'当前设备为{device}')
    return device


# train
def train(net, train_iter, valid_iter, epochs, learning_rate, device, scaler_model,inv = True,conv = True,feature=True):
    train_loss_list = []
    valid_loss_list = []
    bast_loss = np.inf
    # MSE 均方差
    loss_function = nn.MSELoss()  # 定义损失函数
    # MAE 均绝对值误差
    # torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器
    net.to(device)
    loss_function.to(device)
    for epoch in range(epochs):
        net.train()
        train_bar = tqdm(train_iter)
        train_loss = 0  # 均方误差
        for x_train, _, y_train, x_mask, _, y_mask in train_bar:
            optimizer.zero_grad()
            # 将数据放到gpu
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            x_mask = x_mask.to(device)

            y_train_pred = net(x_train, x_mask,inv=inv,conv = conv,feature = feature).to(device)
            y_train_pred = y_train_pred[:, :, -y_train.shape[1]:]
            loss = loss_function(y_train_pred, y_train).sum()
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            train_bar.desc = f'train epoch[{epoch + 1}/{epochs}] loss:{loss}'
            train_loss_list.append([epoch, float(loss.detach().cpu())])
            train_loss += loss
        print(f'train epoch[{epoch + 1}/{epochs}] all_loss:{train_loss}')
        #  评估阶段使用验证集valid_iter
        net.eval()
        with torch.no_grad():
            valid_loss = 0  # 均方误差
            for x_valid, _, y_valid, x_mask, _, y_mask in valid_iter:
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                x_mask = x_mask.to(device)

                y_train_pred = net(x_valid, x_mask,inv=inv,conv = conv,feature = feature).to(device)

                y_train_pred = y_train_pred.to(device)
                loss_single = loss_function(y_train_pred, y_valid).sum()
                valid_loss += loss_single
                valid_loss_list.append([epoch, float(loss_single.detach().cpu())])
                # mae_loss += loss_mae
            print(f'valid epoch[{epoch + 1}/{epochs}] all_loss:{valid_loss}')
            if bast_loss > valid_loss:
                # 训练完毕之后保存模型
                torch.save(net, f'../output/{save_path}/diy_model_mse_{son_dict}.pth')  # 保存模型
                bast_loss = valid_loss
        # subplot_figure(model, dataset_valid, scaler_model, device, 7)
    return net, train_loss_list, valid_loss_list


# 论文形式
def subplot_figure(model, data_loader, scaler_model, device, features_size, save=None):
    model.to(device)
    model.eval()
    x, _, y, x_mask, _, y_mask = data_loader[-2:-1]
    x_origin = scaler_model.inverse_transform(x.reshape(-1, features_size))[:]  # 前96个值
    y_origin = scaler_model.inverse_transform(y.reshape(-1, features_size))[:]  # 真实的后336个值
    y_predict = scaler_model.inverse_transform(model(x.to(device),x_mask.to(device))[:, -y.shape[1]:, :].detach().cpu().reshape(-1, features_size).numpy())[:]
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
    if save is not None:
        f.savefig(f'../output/{save}/Variable_prediction_for_a_single_sample.svg', dpi=3000, bbox_inches='tight')
        # 将预测结果进行保存，方便后面画图
        features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        df = pd.DataFrame(y_predict,columns=features)
        df.to_csv(f'../output/{save}/y_predict.csv', index=False)


# predict
def predict(net, test_iter,device,inv,conv,feature):
    print('开始预测')
    # MSE 均方差
    MSEloss_function = nn.MSELoss()  # 定义损失函数
    # MAE 均绝对值误差
    L1loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器

    net.to(device)
    MSEloss_function.to(device)
    L1loss_function.to(device)
    net.eval()
    with torch.no_grad():
        mse_loss = 0 # 均方误差
        mae_loss = 0
        mse_loss_list = []
        mae_loss_list = []
        for x_test, _, y_test,x_mask, _, y_mask in test_iter:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            x_mask = x_mask.to(device)
            y_test_pred = net(x_test,x_mask,inv = inv,conv = conv,feature = feature)  # 没有强制教学
            y_test_pred = y_test_pred.to(device)
            y_test_pred = y_test_pred[:, -y_test.shape[1]:, :]
            loss_mse = MSEloss_function(y_test_pred, y_test).sum()
            loss_mae = L1loss_function(y_test_pred, y_test).sum()
            mse_loss_list.append(float(loss_mse.detach().cpu()))
            mae_loss_list.append(float(loss_mae.detach().cpu()))
            mse_loss += loss_mse
            mae_loss += loss_mae
        print(f'test all_mse_loss:{mse_loss},all_mae_loss:{mae_loss}')
    return mse_loss_list,mae_loss_list


if __name__ == '__main__':
    class configs():
        seq_len = 96
        model_len = 11
        d_model = 512
        embed = 512
        pred_len = 96
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
        out_channels = pred_len
        activation = 'gelu'


    config = configs()
    #  获取数据
    root_path = 'dataset'
    timestep = 1  # 数据步长
    batch_size = 256  # 批量大小
    epochs = 1  # 轮次
    learning_rate = 0.000001  # 学习率
    features_size = 7  # 数据特征维度
    inv = False
    conv = True
    feature = True
    type_train = True
    save_path = 'Ablation_experiment'
    son_dict = 'featrure'
    scaler_model = MinMaxScaler()
    dataset_train, dataset_valid, dataset_test = data_utils().data_process('../dataset', config.seq_len,
                                                                           config.pred_len, timestep, scaler_model)
    # 将数据载入到dataloader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)
    # 获得设备
    device = get_device()  # 获得设备
    # 获得模型
    model = Model(config)
    if type_train:
        # 模型训练net, train_iter, valid_iter, epochs, device, scaler_model
        model, train_loss_list, valid_loss_list = train(model, train_loader, valid_loader, epochs, learning_rate,
                                                        device, scaler_model,inv = inv,conv = conv,feature = feature)
        pd.DataFrame(train_loss_list).to_csv(f'../output/{save_path}/train_loss_list_{son_dict}.csv', index=False, header=False)
        pd.DataFrame(valid_loss_list).to_csv(f'../output/{save_path}/valid_loss_list_{son_dict}.csv', index=False, header=False)

    model = torch.load(f'../output/{save_path}/diy_model_mse_{son_dict}.pth')
    # 模型预测
    mse_loss_list,mae_loss_list =predict(model, test_loader,device,inv = inv,conv = conv,feature = feature)
    mse_loss_list = pd.Series(mse_loss_list)
    mae_loss_list = pd.Series(mae_loss_list)
    # 消融实验求解标准差和均值
    print(f'mse:mean:{mse_loss_list.mean()},std:{mse_loss_list.std()}')
    print(f'mae:mean:{mae_loss_list.mean()},std:{mae_loss_list.std()}')
    pd.Series(mse_loss_list).to_csv(f'../output/{save_path}/mse_loss_list_{son_dict}.csv', index=False, header=False)
    pd.Series(mae_loss_list).to_csv(f'../output/{save_path}/mae_loss_list_{son_dict}.csv', index=False, header=False)

    # 画图
    # plot_figure(model, dataset_test, scaler_model, device, 7)
    subplot_figure(model, dataset_test, scaler_model, device, 7,save=save_path)
