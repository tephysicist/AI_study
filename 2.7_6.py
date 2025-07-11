import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# здесь объявляйте класс ClassModel
class ClassModel(nn.Module):
    def __init__(self, n_inputs, n_inn, n_outputs):
        super().__init__()
        self.layer1 = nn.Linear(in_features=n_inputs, out_features=n_inn)
        self.layer2 = nn.Linear(in_features=n_inn, out_features=n_outputs)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def act_out(x):
    return torch.where(x < 0, torch.tensor(0), torch.tensor(1))


np.random.seed(1)
torch.manual_seed(1)

# обучающая выборка: x_train - входные значения; y_train - целевые значения
x_train = torch.tensor([(7.7, 2.3), (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), (6.1, 1.2), (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5), (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), (6.0, 1.5), (5.9, 1.8)])
y_train = torch.FloatTensor([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1])




model = ClassModel(2, 3, 1)# здесь создавайте модель
model.train()# переведите модель в режим обучения




total = x_train.size(0) # размер обучающей выборки
N = 1000 # число итераций алгоритма SGD




optimizer = optim.Adam(params=model.parameters(), lr=0.01)# задайте оптимизатор Adam с шагом обучения lr=0.01
loss_func = torch.nn.BCEWithLogitsLoss() # сформируйте функцию потерь (бинарную кросс-энтропию) с помощью класса nn.BCEWithLogitsLoss




for _ in range(N):
    k = np.random.randint(0, total)
    predict = model(x_train[k]) # пропустите через модель k-й образ выборки x_train и вычислите прогноз predict
    loss = loss_func(predict, y_train[k]) # вычислите значение функции потерь и сохраните результат в переменной loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # выполните один шаг градиентного спуска так, как это было сделано в предыдущем подвиге




model.eval() # переведите модель в режим эксплуатации
i = 0
for _x in x_train:
    with torch.no_grad():
        fpredict = act_out(model(_x)) 
        s += (y_train[0] == fpredict).int().float()
        i += 1# прогоните через модель обучающую выборку и подсчитайте долю верных классификаций
Q = s / len(y_train) # результат (долю верных классификаций) сохраните в переменной Q (в виде вещественного числа, а не тензора)
