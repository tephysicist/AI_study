import torch
import torch.nn as nn
import torchvision.transforms as tfs
# import torchvision.transforms.v2 as tfs_v2 - недоступен на Stepik


# здесь объявляйте класс ToDtypeV1
class ToDtypeV1(nn.Module):
    def __init__(self, dtype, scale=False):
        super().__init__()
        self.dtype = dtype
        self.scale = scale
        
    def forward(self, x):
        x = x.to(self.dtype)
        if self.scale and self.dtype in (torch.float16, torch.float32, torch.float64):
            x_min = x.min()
            x_max = x.max()
            x = (x - x_min)/x_max
        return x 



H, W = 128, 128
img_orig = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8) # тензор в программе не менять


img_mean =torch.mean(img_orig.float(), [1, 2]) # средние для каждого цветового канала (первая ось)
img_std = torch.std(img_orig.float().flattern(1, 2), dim=1) for i in range(3)] # стандартное отклонение для каждого цветового канала (первая ось)


# здесь продолжайте программу
tr = tfs.Compose([tfs.ToDtypeV1(dtype=torch.float32, scale=False), tfs.Normalize(mean=[img_mean], std=[img_std])])
img = tr(img_orig)
