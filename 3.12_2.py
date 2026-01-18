import torch
import torch.nn as nn

# здесь объявляйте класс модели (обязательно до тензоров)
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2, 2),
                   stride=(2, 2), padding=0, output_padding=0,
                   groups=1, bias=True, dilation=1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        )
 
    def forward(self, x, y):
        x = self.transpose(x)
        u = torch.cat([x, y], dim=1)
        u = self.block(u)
        return u

# тензоры x, y в программе не менять
batch_size = 2
x = torch.rand(batch_size, 32, 32, 32)
y = torch.rand(batch_size, 16, 64, 64)

# здесь продолжайте программу
model = DecoderBlock()
model.eval()
with torch.no_grad():
    out = model(x, y)
