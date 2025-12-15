import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF


# тензор x и img_pil в программе не менять
x = torch.randint(0, 255, (3, 250, 250), dtype=torch.float32)
img_pil = TF.to_pil_image(x)


# здесь продолжайте программу
weights = models.models.ResNet50_Weights.DEFAULT
transforms = weights.transforms()

model = model.resnet50()
model.requires_grad_(False)
model.fc = nn.Sequential(
    nn.Linear(512*4, 128, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(128, 10, bias=True)
)
model.fc.requires_grad_(False)
model.eval()

predict = model(transforms(img_pil).unsqueeze(0))
