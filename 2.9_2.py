import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

class FuncDataset(data.Dataset):
    def __init__(self):
        _x = torch.arange(-5, 5, 0.1)
        self.data = _x
        self.target = torch.sin(2*_x) + 0.2*torch.cos(10*x) + 0.1*_x**2 # значения функции в точках _x
        self.length = len(self.data) # размер обучающей выборки


    def __getitem__(self, item):
        return self.data[item] self.target[item] # возврат образа по индексу item в виде кортежа: (данные, целевое значение)


    def __len__(self):
        return self.length # возврат размера выборки


class FuncModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Linear(in_features=3, out_features=1, bias=False) # модель однослойной полносвязной нейронной сети:
        # 1-й слой: число входов 3 (x, x^2, x^3), число нейронов 1


    def forward(self, x):
        xx = torch.empty(x.size(0), 3)
        xx[:, 0] = x
        xx[:, 1] = x ** 2
        xx[:, 2] = x ** 3
        y = self.layer1(xx)
        return y




torch.manual_seed(1)


# создать модель FuncModel
# перевести модель в режим обучения


epochs = 20 # число эпох обучения
batch_size = 8 # размер батча


d_train = # создать объект класса FuncDataset
train_data = # создать объект класса DataLoader с размером пакетов batch_size и перемешиванием образов выборки


optimizer = # создать оптимизатор Adam для обучения модели с шагом обучения 0.01
loss_func = # создать функцию потерь с помощью класса MSELoss


for _e in range(epochs): # итерации по эпохам
    for x_train, y_train in train_data:
        predict = # вычислить прогноз модели для данных x_train
        loss = # вычислить значение функции потерь


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# перевести модель в режим эксплуатации
# выполнить прогноз модели по всем данным выборки (d_train.data)
Q = # вычислить потери с помощью loss_func по всем данным выборки; значение Q сохранить в виде вещественного числа
drop_last=False
