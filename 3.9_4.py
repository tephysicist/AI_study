import torch
from torchvision import models
import torchvision.transforms.functional as TF


x = torch.randint(0, 255, (3, 128, 128), dtype=torch.float32)  
img_pil = TF.to_pil_image(x)

# здесь продолжайте программу
weights = models.ResNet18_Weights.DEFAULT
transforms = weights.transforms()
cats = weights.meta['categories']

model = models.resnet18()
model.eval()

img = transforms(img_pil).unsqueeze()
results = model(img).squeeze()
res = results.softmax(dim=0).sort(descending=True)

for i in res[1][:4]:
    print(cats[i])
