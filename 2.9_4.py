import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

class DiabetDataset(data.Dataset, _global_var_data_x, _global_var_target):
    def __init__(self):
        self.data = _global_var_data_x #_global_var_data_x - набор образов выборки размерностью (442, 10);
        self.target = _global_var_target # _global_var_target - целевые значения (уровень сахара) размерностью (442, 1)
        self.length = self.data.size(0) # размер обучающей выборки

    def __getitem__(self, item):
        return self.data[item], self.target[item] # возврат образа по индексу item в виде кортежа: (данные, целевое значение)


    def __len__(self):
        return self.length # возврат размера выборки

class SugarPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=10, out_features=64, bias=True)
        self.layer2 = nn.Linear(in_features=64, out_features=1, bias=True)


    def forward(self, x):
        y = torch.tanh(self.layer1(x))
        z = self.layer2(y)
        return z

model = SugarPred() # создать модель FuncModel
model.train() # перевести модель в режим обучения

epochs = 10 # число эпох обучения
batch_size = 8 # размер батча

d_train = DiabetDataset() # создать объект класса FuncDataset
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
