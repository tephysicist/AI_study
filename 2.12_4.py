import torch
import torch.nn as nn
import torch.nn.functional as F

# здесь продолжайте программу
class Mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=13, out_features=32, bias=True)
        self.layer2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.layer3 = nn.Linear(in_features=16, out_features=3, bias=True)


    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x3

model = Mynn()
st = model.state_dict()
torch.save(st, 'func_nn.tar')
