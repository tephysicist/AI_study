import torch
from torchvision import models
import torchvision.transforms.functional as TF

# тензор x и img_pil в программе не менять
x = torch.randint(0, 255, (3, 128, 128), dtype=torch.float32) 
img_pil = TF.to_pil_image(x)

# здесь продолжайте программу
weights = models.ResNet50_Weights.DEFAULT
transforms = weights.transforms()

inp_img = transforms(img_pil)
