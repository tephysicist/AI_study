import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

# здесь продолжайте программу
model = nn.Sequential( # input [batch_size, 1, 32, 32]
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True), # output [batch_size, 32, 32, 32]
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),  # output [batch_size, 32, 16, 16]
    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True), # output [batch_size, 16, 16, 16]
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False), # output [batch_size, 16, 8, 8]
    nn.Flatten(),
    nn.Linear(1024, 1, bias=True)
        )

d_train, d_test = data.random_split(ds, [0.7, 0.3])  # ds size [256, 1, 32, 32]
train_data = data.DataLoader(d_train, batch_size=16, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

epochs = 2
optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.01)
loss_func = nn.BCEWithLogitsLoss()

model.train()
for _e in range(epochs): # итерации по эпохам
    for x_train, y_train in train_data:
        predict = model(x_train) # вычислить прогноз модели для данных x_train
        loss = loss_func(predict, y_train.unsqueeze(-1)) # вычислить значение функции потерь

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval() # перевести модель в режим эксплуатации
x_test, y_test = next(iter(test_data))
with torch.no_grad():
    p = model(x_test)
    Q = torch.sum(torch.sign(p.flatten()) == (2 * y_test.flatten() - 1)).item()

Q /= len(d_test)
