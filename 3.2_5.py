import torch
import torch.nn as nn

C = 3                   # число каналов
H, W = 17, 19           # размеры изображения: H - число строк; W - число столбцов
kernel_size = (5, 5)    # размер ядра по осям (H, W)
stride = (1, 1)         # шаг смещения ядра по осям (H, W)
padding = (2, 2)        # размер нулевой области вокруг изображения (число строк и столбцов с каждой стороны)

batch_size = 8
x = torch.randint(0, 255, (batch_size, C, H, W), dtype=torch.float32) # тензор x в программе не менять

# здесь продолжайте программу
layer_nn = torch.nn.Conv2d(in_channels=C, out_channels=5, kernel_size=kernel_size, stride=stride, padding=padding)
t_out = layer_nn(x)
