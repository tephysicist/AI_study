import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


model = nn.Sequential(
	nn.Linear(3, 1, bias=True),
) # пропишите модель нейронной сети


# создание обучающей выборки
_x = torch.arange(-5, 5, 0.1)
data_y = torch.sin(2 * _x) + 0.2 * torch.cos(10 * _x) + 0.1 * _x ** 2


_x.unsqueeze_(-1)
data_x = torch.cat([_x, _x ** 2, _x ** 3], dim=1)
ds = data.TensorDataset(data_x, data_y)


epochs = 20 # число эпох обучения
batch_size = 8 # размер батча


train_data = data.DataLoader(ds, batch_size=batch_size, shuffle=True) # создать объект класса DataLoader для датасета ds с размером пакетов batch_size и перемешиванием образов выборки


optimizer = optim.RMSprop(params=model.parameters(), lr=0.01) # создать оптимизатор RMSprop для обучения модели с шагом обучения 0.01
loss_func = nn.MSELoss() # создать функцию потерь с помощью класса MSELoss

model.train()
for _e in range(epochs): # итерации по эпохам
    for x_train, y_train in train_data:
        predict = model(x_train)  # вычислить прогноз модели для данных x_train
        loss = loss_func(predict, y_train.unsqueeze(-1)) # вычислить значение функции потерь

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


model.eval() # перевести модель в режим эксплуатации
predict = model(data_x) # выполнить прогноз модели по всем данным выборки (d_train.data)
Q = loss_func(predict, data_y.unsqueeze(-1)).item() # вычислить потери с помощью loss_func по всем данным выборки; значение Q сохранить в виде вещественного числа
