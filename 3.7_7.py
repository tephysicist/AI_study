import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        y = self.bn(x)
        return nn.functional.relu(y + x)


batch_size = 4
x = torch.rand(batch_size, 256, 16, 16)  # тензор x в программе не менять

# здесь продолжайте программу
model_bn = CNN()

model_bn.eval()
y = model_bn(x)
