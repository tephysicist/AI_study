import torch
import torch.nn as nn
import torch.nn.functional as F

# здесь продолжайте программу
class Mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=48, out_features=32, bias=False)
        self.layer2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.out = nn.Linear(in_features=16, out_features=10, bias=True)


    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = self.out(x2)
        return x3

x = torch.ones(48) # тензор в программе не менять
# здесь продолжайте программу
model = Mynn()
state_dict = torch.load('toy_nn.tar', weights_only=True)
model.load_state_dict(state_dict)
predict = model(x)
