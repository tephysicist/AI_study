import torch
import torch.nn as nn

# здесь объявляйте класс модели
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        y = self.bb(x)
        return y + x

batch_size = 8
x = torch.rand(batch_size, 64, 32, 32)  # тензор x в программе не менять

# здесь продолжайте программу
model_bb = CNN()

model_bb.eval()
y = model_bb(x)
