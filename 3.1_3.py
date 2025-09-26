import torch

C = 3                   # число каналов
H, W = 16, 16           # размеры изображения: H - число строк; W - число столбцов
kernel_size = (3, 3)    # размер ядра по осям (H, W)
stride = (1, 1)         # шаг смещения ядра по осям (H, W)
padding = 0             # размер нулевой области вокруг изображения (число строк и столбцов с каждой стороны)


H_out = int((H + 2 * padding - kernel_size[0]) / stride[0] + 1)
W_out = int((W + 2 * padding - kernel_size[1]) / stride[1] + 1)


x_img = torch.randint(0, 255, (C, H, W), dtype=torch.float32) # тензоры x_img и kernel
kernel = torch.rand((C, ) + kernel_size) # в программе не менять


predict = torch.empty(H_out, W_out, dtype=torch.float32)


# здесь продолжайте программу
