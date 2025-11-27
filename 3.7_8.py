import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256)
        )

        self.sc = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        y1 = self.bn(x)
        y2 =self.sc(x)
        return nn.functional.relu(y1 + y2)


batch_size = 4
x = torch.rand(batch_size, 128, 16, 16) # тензор x в программе не менять


# здесь продолжайте программу
model_bn = BottleneckBlock()
model_bn.eval()

y = model_bn(x)
