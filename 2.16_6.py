import torch
import torch.nn as nn

# здесь объявляйте класс модели
class ModelNN(nn.Module):
    def __init__():
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64)
        )
        self.out_1 = nn.Sequential(
            nn.Linear(64, 10, bias=True),
            nn.Sigmoid()
        )
        self.out_2 = nn.Sequential(
            nn.Linear(64, 32, bias=True),
            nn.Tanh()
        )
    def forward(x):
        x = self.inp(x)
        y = self.out_1(x)
        t = self.out_2(x)
        return y, t

batch_size=28
x = torch.rand(batch_size, 32) # тензор x в программе не менять

model = ModelNN()
model.eval()
predict_y, predict_t = model(x)
