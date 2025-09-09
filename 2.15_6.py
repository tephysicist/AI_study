import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CancerNN(nn.Module):
    def __init__(self, inputs, layer1neurons, layer2neurons, outneurons):
        super().__init__()
        self.layer1 = nn.Linear(in_features=inputs, out_features=layer1neurons, bias=False)
        self.layer2 = nn.Linear(in_features=layer1neurons, out_features=layer2neurons, bias=False)
        self.layer3 = nn.Linear(in_features=layer2neurons, out_features=outneurons)
        self.bn = nn.BatchNorm1d(num_hidden) # output after 1st and 2nd layer has dimension 1 = BatchNorm1d
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.bn(x)
        x = F.relu(self.layer2(x))
        x = self.bn(x)
        x = self.layer3(x)
        return x

model = CancerNN(inputs=30, layer1neurons=32, layer2neurons=20 outneurons=1)

ds = data.TensorDataset(_global_var_data_x, _global_var_target.float())
