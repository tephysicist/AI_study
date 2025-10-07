import torch
import torch.nn as nn


# здесь объявляйте класс модели
class CNN():
    def __init__():
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



# тензоры data_img, data_x в программе не менять
batch_size = 32
data_img = torch.rand(batch_size, 3, 16, 16)
data_x = torch.rand(batch_size, 12)


# здесь продолжайте программу
