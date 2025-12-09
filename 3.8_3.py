import torch
import torch.nn as nn


# здесь объявляйте классы BasicBlock1 и BasicBlock2
class BasicBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            #nn.ReLU(inplace=True)
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(128)
        )
    def forward(self, x):
        y = self.bb(x)
        return nn.ReLU(y + self.layer(x))

class BasicBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True)
        )
    def forward(self, x):
        y = self.bb(x)
        return nn.ReLU(y + x)


batch_size = 8
x = torch.rand(batch_size, 3, 32, 32) # тензор x в программе не менять


# здесь продолжайте программу
model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1, dilation=1, return_indices=False, ceil_mode=False),
    BasicBlock1(),
    BasicBlock2(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, 10, bias=True)
)

model.eval()
predict = model(x)
