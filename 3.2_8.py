import torch
import torch.nn as nn

C = 24         # число входных каналов
H, W = 128, 86 # размеры карт признаков
x = torch.randint(0, 255, (C, H, W), dtype=torch.float32) # тензор x в программе не менять

layer = nn.MaxPool2d(kernel_size=(3, 4), stride=(2, 1), padding=0, dilation=1, return_indices=False, ceil_mode=False)

t_out = layer(x)
