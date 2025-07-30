import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

class DigitDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x  # тензор размерностью (178, 13), тип float32
        self.target = _global_var_target  # тензор размерностью (178, ), тип int64 (long)


        self.length = self.data.size(0) # размер выборки len(self.data)
        self.categories = ['class_0', 'class_1', 'class_2'] # названия классов


    def __getitem__(self, item):
        return self.data[item], self.target[item] # возврат образа по индексу item в виде кортежа: (данные, целевое значение)


    def __len__(self):
        return self.length # возврат размера выборки

class DigitClassModel(nn.Module):
    def __init__(self, in_features=64, out_features=10):
        super().__init__()
        # модель нейронной сети из двух полносвязных слоев:
        self.layer1 = nn.Linear(in_features=in_features, out_features=32, bias=True) # 1-й слой: число входов in_features, число нейронов 16
        self.layer2 = nn.Linear(in_features=32, out_features=16, bias=True) # 2-й слой: число нейронов out_features
        self.layer3 = nn.Linear(in_features=16, out_features=out_features, bias=True)


    def forward(self, x):
        # тензор x пропускается через 1-й слой
        x1 = torch.relu(self.layer1(x)) # через функцию активации torch.relu()
        x2 = torch.relu(self.layer2(x1)) # через второй слой
        x3 = self.layer3(x2)
        return x3 # полученный (вычисленный) тензор возвращается
