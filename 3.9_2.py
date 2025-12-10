import torch
from torchvision import models

weights = models.ResNet50_Weights.DEFAULT
cats = weights.meta['categories']

print(cats[7])
