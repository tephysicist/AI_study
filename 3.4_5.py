import torch
import torchvision.transforms as tfs
import torch.nn as nn

# здесь продолжайте программу
model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False)
        )

tr = tfs.Compose([
    tfs.Resize(224),
    tfs.ToTensor()
])

img = tr(img_pil)

model.eval()
out = model(img.unsqueeze(0))
