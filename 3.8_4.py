import torch
import torch.nn as nn


class BottleneckBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256)
        )
    def forward(self, x):
        y = self.bn(x)
        return nn.functional.relu(y + self.layer(x))

class BottleneckBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256)
        )
    def forward(self, x):
        y = self.bn(x)
        return nn.functional.relu(y + x)

batch_size = 8
x = torch.rand(batch_size, 3, 32, 32) # тензор x в программе не менять


# здесь продолжайте программу
model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1, dilation=1, return_indices=False, ceil_mode=False),
    BottleneckBlock1(),
    BottleneckBlock2(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64*4, 10, bias=True)
)

model.eval()
predict = model(x)
