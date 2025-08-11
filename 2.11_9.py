from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
# import torchvision.transforms.v2 as tfs_v2 - недоступен на Stepik


# здесь объявляйте класс AddNoise
class AddNoise(nn.Module):
    def __init__(self, volume):
        super().__init__()
        self.volume = volume
        
    def forward(self, x):
        return x.to(torch.float32) + torch.randn_like(x) * self.volume


img_pil = Image.new(mode="RGB", size=(128, 128), color=(0, 128, 255))


# здесь продолжайте программу
tr = tfs.Compose([tfs.ToTensor(), tfs.AddNoise(0.1)])
img = tr(img_pil)
