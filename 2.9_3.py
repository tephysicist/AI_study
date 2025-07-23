import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class FuncDataset(data.Dataset):
    def __init__(self):
        _range = torch.arange(-3, 3, 0.1)
        self.data = torch.tensor([(_x, _y) for _x in _range for _y in _range])
        self.target = self._func(self.data)
        self.length = len(_range) # размер обучающей выборки


    @staticmethod
    def _func(coord):
        _x, _y = coord[:, 0], coord[:, 1]
        return torch.sin(2 * _x) * torch.cos(3 * _y) + 0.2 * torch.cos(10 * _x) * torch.sin(8 * _x) + 0.1 * _x ** 2 + 0.1 * _y ** 2

    def __getitem__(self, item):
        return self.data[item], self.target[item] # возврат образа по индексу item в виде кортежа: (данные, целевое значение)

    def __len__(self):
        return self.length # возврат размера выборки




class FuncModel(nn.Module):
    def __init__(self):
        super().__init__()
        # модель однослойной полносвязной нейронной сети:
        self.layer1 = nn.Linear(in_features=6, out_features=1, bias=True) # 1-й слой: число входов 6 (x, x^2, x^3, y, y^2, y^3), число нейронов 1


    def forward(self, coord):
        x, y = coord[:, 0], coord[:, 1]
        x.unsqueeze_(-1)
        y.unsqueeze_(-1)


        xx = torch.empty(coord.size(0), 6)
        xx[:, :3] = torch.cat([x, x ** 2, x ** 3], dim=1)
        xx[:, 3:] = torch.cat([y, y ** 2, y ** 3], dim=1)
        y = self.layer1(xx)
        return y


# здесь продолжайте программу
model = FuncModel() # создать модель FuncModel
model.train() # перевести модель в режим обучения

epochs = 20 # число эпох обучения
batch_size = 16 # размер батча

d_train = FuncDataset() # создать объект класса FuncDataset
train_data = data.DataLoader(d_train, batch_size = batch_size, shuffle=True, drop_last=False) # создать объект класса DataLoader с размером пакетов batch_size и перемешиванием образов выборки


optimizer = optim.RMSprop(params=model.parameters(), lr=0.01) # создать оптимизатор Adam для обучения модели с шагом обучения 0.01
loss_func = torch.nn.MSELoss() # создать функцию потерь с помощью класса MSELoss

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
