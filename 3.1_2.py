import torch

H, W = 16, 16           # размеры изображения: H - число строк; W - число столбцов
kernel_size = (3, 3)    # размер ядра по осям (H, W)
stride = (1, 1)         # шаг смещения ядра по осям (H, W)
padding = 0             # размер нулевой области вокруг изображения (число строк и столбцов с каждой стороны)

H_out = int((H + 2 * padding - kernel_size[0]) / stride[0] + 1)
W_out = int((W + 2 * padding - kernel_size[1]) / stride[1] + 1)

x = torch.randint(0, 255, (H, W), dtype=torch.float32) # входное изображение
kernel = torch.rand(kernel_size) # ядро

predict = torch.empty(H_out, W_out, dtype=torch.float32)
for i in range(0, H_out):
    for j in range(0, W_out):
        predict[i, j] = torch.sum(x[i*stride[0]:kernel_size[0]+i*stride[0],
                                  j*stride[1]:kernel_size[1]+j*stride[1]] * kernel)
