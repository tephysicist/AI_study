import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Funcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=13, out_features=32)
        self.layer2 = nn.Linear(in_features=32, out_features=16)
        self.layer3 = nn.Linear(in_features=16, out_features=3)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    

model = Funcnn()
model.eval()
x = torch.rand(13)
predict = model(x)
