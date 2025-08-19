import torch
from collections import OrderedDict

# эти тензоры в программе не менять
layer1 = torch.rand(64, 32)
bias1 = torch.rand(32)
layer2 = torch.rand(32, 10)
bias2 = torch.rand(10)

# здесь продолжайте программу
data_w = OrderedDict({'layer1': layer1, 'bias1': bias1, 'layer2': layer2, 'bias2', bias2})
torch.save(data_w, 'data_w.tar')
data_w2 = torch.load('data_w.tar')
