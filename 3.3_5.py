import torch
import torch.nn as nn


# здесь объявляйте класс модели
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),
            nn.Flatten()
        )
        self.net2 = nn.Sequential(
            nn.Linear(12, 64, bias=False),
            nn.Sigmoid(),
            nn.BatchNorm1d(64)
        )
        # For example, if input is [batch_size, 3, 16, 16], after net1:
        # - Conv2d: [batch_size, 16, 16, 16]
        # - MaxPool2d: [batch_size, 16, 8, 8]
        # - Conv2d: [batch_size, 32, 8, 8]
        # - MaxPool2d: [batch_size, 32, 4, 4]
        # - Flatten: [batch_size, 32 * 4 * 4] = [batch_size, 512] 512+64
        output = Linear(576, 10)
    def forward(self, x1, x2):
        x1 = self.net1(x1)
        x2 = self.net2(x2)
        x = torch.cat((x1, x2), dim=1)  # Concatenate along dim=1: [batch_size, 2048 + 64]
        return self.output(x)



# тензоры data_img, data_x в программе не менять
batch_size = 32
data_img = torch.rand(batch_size, 3, 16, 16)
data_x = torch.rand(batch_size, 12)


# здесь продолжайте программу
model = CNN()
model.eval()
predict = model(data_img, data_x)
