import torch
import torch.nn as nn
import torch.optim as optim

# здесь продолжайте программу
class Mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=10, out_features=8, bias=False)
        self.layer2 = nn.Linear(in_features=8, out_features=4, bias=True)
        self.out = nn.Linear(in_features=4, out_features=6, bias=True)


    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.out(x)
        return x

model = Mynn()
opt = optim.RMSprop(params=model.parameters(), lr=0.05)

st = torch.load('nn_data_state.tar', weights_only=True)
model.load_state_dict(st['model'])
opt.load_state_dict(st['opt'])
