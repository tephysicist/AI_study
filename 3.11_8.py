import torch
import torch.nn as nn


batch_size = 4
H, W = 16, 24
x = torch.rand(batch_size, 3, H, W) # тензор x в программе не менять


# здесь продолжайте программу
layer = nn.ConvTranspose2d(in_channels=3, out_channels=2, kernel_size=(3, 3),
                   stride=(2, 2), padding=0, output_padding=0,
                   groups=1, bias=True, dilation=1, padding_mode='zeros')

out = layer(x)
