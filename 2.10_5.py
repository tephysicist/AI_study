import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

class DigitDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x  # тензор размерностью (178, 13), тип float32
        self.target = _global_var_target  # тензор размерностью (178, ), тип int64 (long)
        self.length = self.data.size(0) # размер выборки len(self.data)


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

model = DigitClassModel() # создать модель IrisClassModel с числом входов 4 и числом выходов 3
model.train() # перевести модель в режим обучения


epochs = 10 # число эпох обучения
batch_size = 12 # размер батча

d_train = DigitDataset() # создать объект класса IrisDataset
train_data = data.DataLoader(d_train, batch_size = batch_size, shuffle=True, drop_last=False) # создать объект класса DataLoader с размером пакетов batch_size и перемешиванием образов выборки


optimizer = optim.Adam(params=model.parameters(), lr=0.01) # создать оптимизатор Adam для обучения модели с шагом обучения 0.01 optim.Adam(params=model.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss() # создать функцию потерь с помощью класса CrossEntropyLoss (используется при многоклассовой классификации)

for _e in range(epochs): # итерации по эпохам
    for x_train, y_train in train_data:
        predict = model(x_train) # вычислить прогноз модели для данных x_train
        loss = loss_func(predict, y_train) # вычислить значение функции потерь


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval() # перевести модель в режим эксплуатации
predict = model(d_train.data) # выполнить прогноз модели по всем данным выборки
p = torch.argmax(predict, dim=1)
Q = torch.mean((d_train.target == p).float()).item() # вычислить долю верных классификаций (сохранить, как вещественное число, а не тензор)
