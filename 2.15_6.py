import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CancerNN(nn.Module):
    def __init__(self, inputs, layer1neurons, layer2neurons, outneurons):
        super().__init__()
        self.layer1 = nn.Linear(in_features=inputs, out_features=layer1neurons, bias=False)
        self.layer2 = nn.Linear(in_features=layer1neurons, out_features=layer2neurons, bias=False)
        self.layer3 = nn.Linear(in_features=layer2neurons, out_features=outneurons)
        self.bn1 = nn.BatchNorm1d(layer1neurons) # output after 1st and 2nd layer has dimension 1 => BatchNorm1d
        self.bn2 = nn.BatchNorm1d(layer2neurons) # output after 1st and 2nd layer has dimension 1 => BatchNorm1d
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.bn1(x)
        x = F.relu(self.layer2(x))
        x = self.bn2(x)
        x = self.layer3(x)
        return x

model = CancerNN(inputs=30, layer1neurons=32, layer2neurons=20, outneurons=1)

batch_size = 16
epochs = 5
ds = data.TensorDataset(_global_var_data_x, _global_var_target.float())
d_train, d_test = data.random_split(ds, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

optimizer = optim.Adam(params=model.parameters(), lr=0.01)
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
