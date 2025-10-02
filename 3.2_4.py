import torch
import torch.nn as nn

C = 5                   # число каналов
H, W = 32, 24           # размеры изображения: H - число строк; W - число столбцов
kernel_size = (5, 3)    # размер ядра по осям (H, W)
stride = (1, 1)         # шаг смещения ядра по осям (H, W)
padding = 0             # размер нулевой области вокруг изображения (число строк и столбцов с каждой стороны)

x = torch.randint(0, 255, (C, H, W), dtype=torch.float32) # тензор x в программе не менять

# здесь продолжайте программу
layer_nn = torch.nn.Conv2d(in_channels=C, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)

t_out = layer_nn(x) # layer_nn(x.unsqueeze(0))
