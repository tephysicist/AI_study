import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TelNN(nn.Module):
    def __init__(self, inputs, layer1neurons, outneurons):
        super().__init__()
        self.layer1 = nn.Linear(in_features=inputs, out_features=layer1neurons, bias=False)
        self.layer2 = nn.Linear(in_features=layer1neurons, out_features=outneurons)
        self.bn = nn.BatchNorm1d(layer1neurons) # output after 1st layer has dimension 1 = BatchNorm1d
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.bn(x)
        x = self.layer2(x)
        return x

model = TelNN(inputs=10, layer1neurons=64, outneurons=1)

batch_size = 16
x = torch.rand(batch_size, 10)  # этот тензор в программе не менять

model.eval()
predict = model(x)
