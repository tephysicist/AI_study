import torch
import torch.nn as nn

x = torch.rand(1, 16, 16)  # тензор x в программе не менять

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),
    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),
    nn.Flatten(),
    nn.Linear(256, 5, bias=True)
)

model.eval()
predict = model(x.unsqueeze(0))
