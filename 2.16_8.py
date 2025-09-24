import torch
import torch.nn as nn

class DeepNetwork(nn.Module):
    def __init__(self, n_hidden_layers):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(11, 32, bias=False),
            nn.ReLU()
        )
        self.layers = nn.ModuleList(
            [nn.Linear(32, 32) for i in range(n_hidden_layers)]
        )
        self.outut = self.input = nn.Sequential(nn.Linear(11, 32, bias=False))
    def forward(self, x):
        x = self.input(x)
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.outut(x)
        return x
            


n = int(input())

batch_size = 18
x = torch.rand(batch_size, 11) # тензор x в программе не менять

model = DeepNetwork(n)
model.eval()
predict = model(x)
