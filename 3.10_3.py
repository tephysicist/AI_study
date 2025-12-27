import torch
from torchvision import models
import torchvision.transforms.functional as TF

x = torch.randint(0, 255, (3, 128, 128), dtype=torch.float32)  
img_pil = TF.to_pil_image(x)

weights = models.ResNet34_Weights.DEFAULT
transforms = weights.transforms()

model = models.resnet34()
model.requires_grad_(False)
model.eval()

img = transforms(img_pil).unsqueeze(0)
results = model(img)
