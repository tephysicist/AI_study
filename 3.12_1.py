import torch
import torch.nn as nn

# здесь объявляйте класс модели (обязательно до тензора x)
class CNV(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.layer3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x = self.bn1(torch.relu(self.layer1(x)))
        y = self.bn2(torch.relu(self.layer2(x)))
        z = self.layer3(y)
        return z, y


x = torch.rand(3, 128, 128) # тензор x в программе не менять

# здесь продолжайте программу
model = CNV()
model.eval()
with torch.no_grad():
    out1, out2 = model(x.unsqueeze(0))
