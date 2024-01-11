import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import torch
from torch import nn, Tensor
import positional_encoder as pe
from model import TimeSeriesForcasting
import inference
from CLASS.data_class import data_utils

# 获得gpu
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'当前设备为{device}')
    return device


# 论文形式
def subplot_figure(model, data_loader, scaler_model, device, features_size, save=None,teaching_focre =False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x, decoder_input,y = data_loader[-2:-1]
        x_origin = scaler_model.inverse_transform(x.reshape(-1, features_size))[:]  # 前96个值
        y_origin = scaler_model.inverse_transform(y.reshape(-1, features_size))[:]  # 真实的后o个时间点
        x = x.to(device)
        y = y.to(device)
        decoder_input = decoder_input.to(device)
        if teaching_focre:
            y_predict = model((x, decoder_input))
        else:
            y_predict = inference.my_inference(model=model, src=x.to(device), forecast_window=y.shape[1],device=device)
        y_predict = scaler_model.inverse_transform(y_predict.detach().cpu().reshape(-1, features_size).numpy())[
                    :]
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
        if save is not None:
            f.savefig(f'output/{save_path}/Variable_prediction_for_a_single_sample.svg', dpi=3000, bbox_inches='tight')
            # 将预测结果进行保存，方便后面画图
            features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
            df = pd.DataFrame(y_predict, columns=features)
            df.to_csv(f'output/{save}/y_predict.csv', index=False)


def train(net, train_iter, valid_iter, epochs, device, scaler_model):
    train_loss_list = []
    valid_loss_list = []
    best_loss = np.inf
    # 获得loss
    loss_function = nn.MSELoss()  # 定义损失函数
    # 获得优化器和学习率策略
    optimizer, scheduler = net.configure_optimizers()
    net.to(device)
    loss_function.to(device)
    for epoch in range(epochs):
        net.train()
        train_bar = tqdm(train_iter)
        train_loss = 0  # 均方误差
        for x_train, decoder_input_train, y_train in train_bar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            decoder_input_train = decoder_input_train.to(device)
            optimizer.zero_grad()
            loss = net.training_step((x_train, decoder_input_train, y_train), loss_function)
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            train_bar.desc = f'train epoch[{epoch + 1}/{epochs}] loss:{loss}'
            train_loss_list.append([epoch, float(loss.detach().cpu())])
            train_loss += loss
        print(f'train epoch[{epoch + 1}/{epochs}] all_loss:{train_loss}')

        net.eval()
        with torch.no_grad():
            valid_loss = 0  # 均方误差
            valid_bar = tqdm(valid_iter)
            for x_valid, decoder_input_valid, y_valid in valid_bar:
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                decoder_input_valid = decoder_input_valid.to(device)
                loss = net.training_step((x_valid, decoder_input_valid, y_valid), loss_function)
                valid_loss += loss
                valid_loss_list.append([epoch, float(loss.detach().cpu())])
                valid_bar.desc = f'valid epoch[{epoch + 1}/{epochs}] loss:{loss}'
                # mae_loss += loss_mae
            print(f'valid epoch[{epoch + 1}/{epochs}] all_loss:{valid_loss}')
            scheduler.step(valid_loss)
            if best_loss > valid_loss:
                # 训练完毕之后保存模型
                torch.save(net, f'output/{save_path}/transformer_model_mae.pth')  # 保存模型
                best_loss = valid_loss
        # 展示效果
        subplot_figure(net, dataset_train, scaler_model, device, features_size)
        # 推理
    return train_loss_list, valid_loss_list




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
        for x_test, _, y_test in test_bar:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_predict = inference.my_inference(model=model, src=x_test.to(device), forecast_window=y_test.shape[1],device=device)
            y_predict = y_predict.to(device)
            loss_mse = MSEloss_function(y_predict, y_test).sum()
            loss_mae = L1loss_function(y_predict, y_test).sum()
            mse_loss_list.append(float(loss_mse.detach().cpu()))
            mae_loss_list.append(float(loss_mae.detach().cpu()))
            mse_loss += loss_mse
            mae_loss += loss_mae
        print(f'test all_mse_loss:{mse_loss},all_mae_loss:{mae_loss}')
    return mse_loss_list,mae_loss_list




if __name__ == '__main__':
    teaching_force = 0.4
    #  获取数据
    root_path = 'dataset'
    input_size = 96  # 输入维度
    output_size = 96  # 预测维度
    timestep = 1  # 数据步长
    batch_size = 256  # 批量大小
    epochs = 150  # 轮次
    learning_rate = 1e-3  # 学习率
    features_size = 7  # 数据特征维度
    dropout = 0.1
    num_layers = 2
    channels = 256
    type_train = False
    save_path = 'transformer_short'
    scaler_model = MinMaxScaler()
    dataset_train, dataset_valid, dataset_test = data_utils().data_process('dataset', input_size, output_size, timestep, scaler_model)
    # 将数据载入到dataloader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)
    # 获得设备
    device = get_device()  # 获得设备
    # 获得模型
    model = TimeSeriesForcasting(
        n_encoder_inputs=features_size, n_decoder_inputs=features_size,
        num_layers=num_layers, channels=channels, lr=learning_rate, dropout=dropout)
    if type_train:
        # 模型训练
        train_loss_list,valid_loss_list = train(model,train_loader,valid_loader,epochs,device,scaler_model)
        pd.DataFrame(train_loss_list).to_csv(f'output/{save_path}/train_loss_list.csv', index=False, header=False)
        pd.DataFrame(valid_loss_list).to_csv(f'output/{save_path}/valid_loss_list.csv', index=False, header=False)
        torch.save(model, f'output/{save_path}/transformer_model_mae_final.pth')  # 保存模型
    else:

        model = torch.load(f'output/{save_path}/transformer_model_mae_final.pth')
        # # 模型预测
        # mse_loss_list,mae_loss_list = predict(model, test_loader,device)
        # pd.Series(mse_loss_list).to_csv(f'output/{save_path}/mse_loss_list.csv', index=False, header=False)
        # pd.Series(mae_loss_list).to_csv(f'output/{save_path}/mae_loss_list.csv', index=False, header=False)

        # 画图
        #plot_figure(model, dataset_test, scaler_model, device, 7)
        subplot_figure(model, dataset_test, scaler_model, device, 7,save=save_path)


