import torch

H, W = 24, 24           # размеры карты признаков: H - число строк; W - число столбцов
kernel_size = (3, 2)    # размер окна для Pooling по осям (H, W)
stride = (2, 2)         # шаг смещения окна по осям (H, W)
padding = 1             # размер нулевой области вокруг карты признаков (число строк и столбцов с каждой стороны)

H_out = int((H + 2 * padding - kernel_size[0]) / stride[0] + 1)
W_out = int((W + 2 * padding - kernel_size[1]) / stride[1] + 1)

x = torch.rand((H, W)) # карта признаков (в программе не менять)

# здесь продолжайте программу
x_p = torch.zeros((H + 2*padding, W + 2*padding))
x_p[padding:H + padding, padding:W + padding] = x

res_pool = torch.empty(H_out, W_out, dtype=torch.float32)

for i in range(0, H_out):
    for j in range(0, W_out):
        res_pool[i, j] = torch.mean(x_p[i*stride[0]:kernel_size[0]+i*stride[0], j*stride[1]:kernel_size[1]+j*stride[1]])
