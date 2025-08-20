import torch
import torch.nn as nn
import torch.optim as optim

# здесь продолжайте программу
class Mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=25, out_features=16, bias=False)
        self.layer2 = nn.Linear(in_features=16, out_features=8, bias=True)
        self.out = nn.Linear(in_features=8, out_features=5, bias=False)


    def forward(self, x):
        x1 = torch.tanh(self.layer1(x))
        x2 = torch.tanh(self.layer2(x1))
        x3 = self.out(x2)
        return x3

model = Mynn()
opt = optim.Adam(params=model.parameters(), lr=0.02)
loss_func = nn.CrossEntropyLoss()
