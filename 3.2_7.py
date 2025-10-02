import torch
import torch.nn as nn

H, W = 32, 25
x = torch.randint(0, 255, (H, W), dtype=torch.float32) # тензор x в программе не менять

layer = torch.nn.MaxPool2d(kernel_size=(3, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False) # lr = nn.MaxPool2d((3, 2))

t_out = layer(x) # lr(x.view(1, 1, H, W))
