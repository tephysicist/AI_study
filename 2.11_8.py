import torch
import torch.nn as nn
import torchvision.transforms as tfs
# import torchvision.transforms.v2 as tfs_v2 - недоступен на Stepik


# здесь объявляйте класс ToDtypeV1
class ToDtypeV1(nn.Module):
    def __init__(self, dtype=torch.float32, scale=False):
        self.dtype = dtype
        self.scale = scale
        
    def forward(self, item):
        #item = item.to(self.dtype)
        norm = tfs.Normalize(mean=[0.5], std=[0.5])
        if (self.dtype == torch.float16 or self.dtype == torch.float32 or self.dtype == torch.float64) and self.scale:
            return item.to(self.dtype)
        else:
            return norm(item) 



H, W = 128, 128
img_orig = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8) # тензор в программе не менять


img_mean =[torch.mean(img_orig[i], dim=0) for i in range(3)] # средние для каждого цветового канала (первая ось)
img_std = [torch.std(img_orig[i], dim=0) for i in range(3)] # стандартное отклонение для каждого цветового канала (первая ось)


# здесь продолжайте программу
tr = tfs.Compose([tfs.ToDtypeV1(dtype=torch.float32, scale=False), tfs.Resize(128), tfs.ToTensor()])
