from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
# dataset
class my_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        # 下标来调用数据
        return self.data[0][idx], self.data[1][idx],self.data[2][idx]
    def __len__(self):
        return len(self.data[0])



class data_utils():
    # 对于每个数据文件都要变化成为对应的x和label
    def get_dataset(self,data, input_size, output_size, timestep):
        data_windows = []
        for index in range(0, len(data) - input_size - output_size, timestep):
            data_windows.append(data[index:index + input_size + output_size])
        # 未划分decoder encoder输入 target
        data_all = torch.tensor(np.array(data_windows)).to(torch.float32)
        encoder_input = data_all[:, :input_size, :]
        label = data_all[:, input_size:, :]
        # 起始符为0
        start_tgt = torch.zeros((encoder_input.shape[0], 1, encoder_input.shape[2]))
        decoder_input = torch.cat((start_tgt, label[:, :-1, :]), dim=1)
        dataset = my_dataset([encoder_input, decoder_input, label])
        return dataset


    def concat_Data(self,Data, scaler_model):
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
    def data_process(self,root_path, input_size, output_size, timestep, scaler_model):
        '''
        :param root_path: 根目录
        :param input_size:输入的维度，默认为96
        :param output_size:每个样本的预测维度，默认为96，后面会改成336
        :param timestep: 时间步，滑动窗口
        '''
        # 获取对应类型的数据
        files = os.listdir(root_path)
        files_csv = sorted([f for f in files if f.endswith('.csv')])
        validation = train = test = None
        for file in files_csv:
            if file.split('_')[0] == 'validation':
                validation = pd.read_csv(os.path.join(root_path, file)).drop('date', axis=1)  # 得到每个文件数据
            elif file.split('_')[0] == 'train':
                train = pd.read_csv(os.path.join(root_path, file)).drop('date', axis=1)  # 得到每个文件数据
            elif file.split('_')[0] == 'test':
                test = pd.read_csv(os.path.join(root_path, file)).drop('date', axis=1)  # 得到每个文件数据
        # 得到了所有数据，开始归一化
        [train, validation, test] = self.concat_Data([train, validation, test], scaler_model)
        dataset_train = self.get_dataset(train, input_size, output_size, timestep)
        dataset_validation = self.get_dataset(validation, input_size, output_size, timestep)
        dataset_test = self.get_dataset(test, input_size, output_size, timestep)
        return dataset_train, dataset_validation, dataset_test


if __name__ == '__main__':
    scaler_model = MinMaxScaler()
    a,b,c = data_utils().data_process('../dataset', 96, 96, 1, scaler_model)
    print(a)

