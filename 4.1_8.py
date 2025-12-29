import torch
import torch.nn as nn

# здесь объявляйте класс модели
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_size = 10
        self.layer1 = nn.Linear(16, self.h_size, bias=True)
        self.layer2 = nn.Linear(self.h_size, 5, bias=True)
    def forward(self, x): # (batch_size, seq_length, d_size)
        n = x.size(1)
        b = x.size(0)
        h = torch.zeros(b, self.h_size)
        
        for i in range(n):
            a = self.layer1(x[:, i, :])
            h = torch.tanh(h + a)
        y = torch.sigmoid(self.layer2(h))
        return y
        
        

batch_size = 8 # размер батча
seq_length = 6 # длина последовательности
in_features = 16 # размер каждого элемента последовательности
x = torch.rand(batch_size, seq_length, in_features)


# здесь объявляйте саму модель и продолжайте программу
model = RNN()
model.eval()
out = model(x)
