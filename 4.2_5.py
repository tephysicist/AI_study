import torch
import torch.nn as nn


# здесь объявляйте класс OutputModule
class OutputModule():
    def __init__():
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.layer = nn.Linear(25, 10, bias=True)
        
    def forward(self, x): 
        return self.layer(self.act(x[0]))


# тензор x в программе не менять
batch_size = 7
seq_length = 5
in_features = 15
x = torch.rand(batch_size, seq_length, in_features)


# здесь продолжайте программу
model = nn.Sequential(
    nn.RNN(input_size=in_features, hidden_size=25, num_layers=1, nonlinearity='tanh', bias=True, batch_first=True),
    OutputModule()
)

model.eval()
out = model(x)
