import torch
import torch.nn as nn


# здесь объявляйте класс OutputToLinear
class OutputToLinear(nn.Module):
    def forward(self, x):
        return x[1][-1]
        

# тензор x в программе не менять
batch_size = 18
seq_length = 21
in_features = 5
x = torch.rand(batch_size, seq_length, in_features)


# здесь продолжайте программу
model = nn.Sequential(
    nn.RNN(input_size=in_features, hidden_size=25, num_layers=2, nonlinearity='tanh', bias=True, batch_first=True),
    OutputToLinear(),
    nn.ReLU(inplace=True),
    nn.Linear(25, 5, bias=True)
)

model.eval()
predict = model(x)
