import os
import pickle

import numpy as np
import pandas as pd
import torch
from dtw import dtw

columns = ["po", "pdmp", "pin"]


class Dataset():
    def __init__(self, individual=1, split_para1=0.7, split_para2=0.2):
        super(Dataset).__init__()
        self.data = None
        self.labels = None
        self.split_point1 = None
        self.split_point2 = None
        self.individual = individual
        self.split_para1 = split_para1
        self.split_para2 = split_para2

    # 序列对齐
    def data_processing(self):

        with open(f"../data/HRD/Individual{self.individual}/data.pkl", 'rb') as file:
            data = pickle.load(file)
        with open(f"../data/HRD/Individual{self.individual}/labels.pkl", 'rb') as file:
            labels = pickle.load(file)

        labels = [x - 1 for x in labels]

        def dtw_time_series(data):
            result_list = []
            reference_seq = min(data, key=lambda arr: arr.shape[0])
            for i in range(len(data)):
                seq = data[i]
                _dtw = dtw(reference_seq, seq)
                len1 = _dtw.N
                len2 = _dtw.M
                if len1 > len2:
                    index1 = _dtw.index2
                    index2 = _dtw.index1
                else:
                    index1 = _dtw.index1
                    index2 = _dtw.index2
                unique_values, indices = np.unique(index1, return_index=True)
                result = index2[indices]
                arr_list = []
                for element in result:
                    arr_list.append(data[i][element])
                    arr = np.array(arr_list)
                result_list.append(arr)

            return result_list

        self.data = dtw_time_series(data)
        self.labels = labels

        self.split_point1 = int(self.split_para1 * len(self.data))
        self.split_point2 = int(self.split_para2 * len(self.data))

    # 返回df格式的三种数据集
    def get_train_data(self):
        train_data = pd.DataFrame({'data': self.data[: self.split_point1], 'labels': self.labels[: self.split_point1]})
        return train_data

    def get_test_data(self):
        test_data = pd.DataFrame({'data': self.data[self.split_point1:self.split_point2],
                                  'labels': self.labels[self.split_point1:self.split_point2]})
        return test_data

    def get_valid_data(self):
        valid_data = pd.DataFrame({'data': self.data[self.split_point2:],
                                   'labels': self.labels[self.split_point2:]})
        return valid_data

    # 将数据处理为字典形式
    def get_dataset_dict(self, input_data):
        # Reshape the data to (samples, time steps, features)
        data = input_data['data'].values
        # 使用 np.stack 将列表中的 ndarrays 堆叠在新的维度上
        tensor_data = np.stack(data, axis=0)
        labels_array = np.array(input_data['labels'])
        transposed_tensor = np.transpose(tensor_data, (0, 2, 1))
        dict = {'samples': transposed_tensor, 'labels': labels_array}
        return dict

    def return_all_class_list(self):
        train_list = []
        test_list = []
        valid_list = []
        for i in range(1, 9):
            train_list.append(torch.load(os.path.join(f"../data\\HRD_all_class\\Individual{str(i)}",
                                                      "train.pt")))
            valid_list.append(torch.load(os.path.join(f"../data\\HRD_all_class\\Individual{str(i)}",
                                                      "val.pt")))
            test_list.append(torch.load(os.path.join(f"../data\\HRD_all_class\\Individual{str(i)}",
                                                     "test.pt")))

        return train_list, test_list, valid_list


if __name__ == '__main__':
    dataset = Dataset()
    train_list, test_list, valid_list = dataset.return_all_class_list()
