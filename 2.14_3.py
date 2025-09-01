import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class FuncModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer = nn.Linear(in_features=in_features, out_features=1) # полносвязный слой: число входов in_features, число нейронов 1

    def forward(self, x):
        return self.layer(x)

torch.manual_seed(1)

epochs = 20 # число эпох обучения
batch_size = 8 # размер батча
N = 6 # порядок модели (N-1)

# формирование обучающей выборки (значений функции)
data_x = torch.arange(-4, 4, 0.01)
data_y = 0.1 * data_x + 0.1 * data_x ** 2 - 0.5 * torch.sin(2 * data_x) + torch.cos(4 * data_x)
data_x.unsqueeze_(-1)
X = torch.cat([data_x ** _n for _n in range(N)], dim=1)
ds = data.TensorDataset(X, data_y)


d_train, d_val = data.random_split(ds, [0.8, 0.2]) #разделите выборку ds на две части в пропорции: 80% и 20%
train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=len(d_val), shuffle=False)


model = FuncModel(N) # most probable N-1
optimizer = optim.RMSprop(params=model.parameters(), lr=0.01) # создайте оптимизатор RMSprop для модели model с шагом обучения 0.01 и weight_decay=10
loss_func = nn.MSELoss() # сформируйте функцию потерь с помощью класса MSELoss


loss_lst_val = []  # список значений потерь при валидации
loss_lst = []  # список значений потерь при обучении


for _e in range(epochs):
    loss_mean = 0
    lm_count = 0
    
    # learning model with loss_mean calculation
    model.train()
    for x_train, y_train in train_data:
        predict = model(x_train) # вычислите прогноз модели для x_train
        loss = loss_func(predict, y_train.unsqueeze(-1)) # вычислите потери для predict и y_train

        # выполните один шаг обучения (градиентного спуска)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # calculating the average value of the loss function over the entire sample 
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean


    # валидация модели
    # переведите модель в режим эксплуатации
    model.eval()
    x_val, y_val = next(iter(train_data_val))


    with torch.no_grad():
        p = model(x_val) #вычислите прогноз модели для x_val
        loss = loss_func(p, y_val.unsqueeze(-1)) вычислите потери для p и y_val
        Q_val = loss.item()


    loss_lst.append(loss_mean)
    loss_lst_val.append(Q_val)


 model.eval() # переведите модель в режим эксплуатации
p = model(data_x)# вычислите прогноз модели по всем данным выборки X
Q = loss_func(p, data_y.unsqueeze(-1)).item() # вычислите потери с помощью loss_func для p и data_y; значение Q сохраните в виде вещественного числа
