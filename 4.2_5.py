import torch
import torch.nn as nn


# здесь объявляйте класс OutputModule
class OutputModule():
    def forward(self, x):
        return


# тензор x в программе не менять
batch_size = 7
seq_length = 5
in_features = 15
x = torch.rand(batch_size, seq_length, in_features)


# здесь продолжайте программу
model = nn.Sequential(
    nn.RNN(input_size=in_features, hidden_size=25, num_layers=1, nonlinearity='tanh', bias=True, batch_first=True),
    OutputModule(),
    nn.ReLU(inplace=True),
    nn.Linear(15, 5, bias=True)
)
