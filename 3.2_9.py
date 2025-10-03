import torch
import torch.nn as nn

batch_size = 8 # размер батча
C = 24         # число входных каналов
H, W = 128, 86 # размеры карт признаков
x = torch.randint(0, 255, (batch_size, C, H, W), dtype=torch.float32) # тензор x в программе не менять

layer = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)

t_out = layer(x)
