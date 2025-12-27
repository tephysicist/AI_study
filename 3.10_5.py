import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF


# тензор x и img_pil в программе не менять
x = torch.randint(0, 255, (3, 250, 250), dtype=torch.float32)
img_pil = TF.to_pil_image(x)


# здесь продолжайте программу
weights = models.ResNet50_Weights.DEFAULT
transforms = weights.transforms()

model = models.resnet50()
model.requires_grad_(True)
model.fc = nn.Sequential(
    nn.Linear(512*4, 100, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(100, 10, bias=True)
)
model.fc.requires_grad_(True)
model.eval()
with torch.no_grad(): predict = model(transforms(img_pil).unsqueeze(0))
