import torch
import torch.nn as nn

class ModelNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_1 = nn.Sequential(
            nn.Linear(7, 12, bias=True),
            nn.Tanh()
        )
        self.inp_2 = nn.Sequential(
            nn.Linear(12, 12, bias=True),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(12, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True)
        )
    def forward(a, b):
        sum = self.inp_1(a) + self.inp_2(b)
        return self.out(sum)



batch_size=12
a = torch.rand(batch_size, 7)
b = torch.rand(batch_size, 12)

model = ModelNN()
model.eval()
predict = model(a, b)
