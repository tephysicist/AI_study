import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

class FuncDataset(data.Dataset):
    def __init__(self):
        _x = torch.arange(-6, 6, 0.1)
        self.data = _x
        self.target = 0.5*_x + torch.sin(2*_x) + 0.1*torch.exp(_x/2) - 12.5 # значения функции в точках _x
        self.length = len(self.data) # размер обучающей выборки

    def __getitem__(self, item):
        return self.data[item], self.target[item] # возврат образа по индексу item в виде кортежа: (данные, целевое значение)

    def __len__(self):
        return self.length # возврат размера выборки


class FuncModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=3, out_features=1, bias=True) # модель однослойной полносвязной нейронной сети:
        # 1-й слой: число входов 3 (x, x^2, x^3), число нейронов 1

    def forward(self, x):
        '''
        xx = torch.empty(x.size(0), 3)
        xx[:, 0] = x
        xx[:, 1] = x ** 2
        xx[:, 2] = x ** 3
        y = self.layer1(xx)
        '''
        x.unsqueeze_(-1)
        xx = torch.cat([x, x ** 2, x ** 3], dim=1)
        y = self.layer1(xx)
        return y

model = FuncModel() # создать модель FuncModel
model.train() # перевести модель в режим обучения

epochs = 20 # число эпох обучения
batch_size = 10 # размер батча

d_train = FuncDataset() # создать объект класса FuncDataset
train_data = data.DataLoader(d_train, batch_size = batch_size, shuffle=True) # создать объект класса DataLoader с размером пакетов batch_size и перемешиванием образов выборки


optimizer = optim.RMSprop(params=model.parameters(), lr=0.1) # создать оптимизатор Adam для обучения модели с шагом обучения 0.01
loss_func = nn.MSELoss() # создать функцию потерь с помощью класса MSELoss

for _e in range(epochs): # итерации по эпохам
    for x_train, y_train in train_data:
        predict = model(x_train) # вычислить прогноз модели для данных x_train
        loss = loss_func(predict, y_train.unsqueeze(-1)) # вычислить значение функции потерь
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval() # перевести модель в режим эксплуатации
predict = model(d_train.data) # выполнить прогноз модели по всем данным выборки (d_train.data)
Q = loss_func(predict, d_train.target.unsqueeze(-1)).item() # вычислить потери с помощью loss_func по всем данным выборки; значение Q сохранить в виде вещественного числа
