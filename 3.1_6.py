import torch

C = 3                   # число каналов
H, W = 16, 12           # размеры изображения: H - число строк; W - число столбцов
kernel_size = (5, 3)    # размер ядра по осям (H, W)
stride = (1, 2)         # шаг смещения ядра по осям (H, W)
padding = 1             # размер нулевой области вокруг изображения (число строк и столбцов с каждой стороны)

bias = torch.rand(1)    # смещение для фильтра (ядра), коэффициент w0
act = torch.tanh        # функция активации нейронов (результатов свертки)

H_out = int((H + 2 * padding - kernel_size[0]) / stride[0] + 1)
W_out = int((W + 2 * padding - kernel_size[1]) / stride[1] + 1)

x_img = torch.randint(0, 255, (C, H, W), dtype=torch.float32) # тензоры x_img и kernel
kernel = torch.rand((C, ) + kernel_size) # в программе не менять

# здесь продолжайте программу
x_img_p = torch.zeros((C, H + 2*padding, W + 2*padding))
x_img_p = x_img_p[:, padding:H+padding, padding:W+padding] = x_img

predict = torch.empty(H_out, W_out, dtype=torch.float32)
for i in range(0, H_out):
    for j in range(0, W_out):
        predict[i, j] = torch.sum(x_img_p[:, i*stride[0]:kernel_size[0]+i*stride[0], j*stride[1]:kernel_size[1]+j*stride[1]] * kernel)

predict = act(predict) #???? + bias act(predict + bias)
