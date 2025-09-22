import torch
import torch.nn as nn

batch_size=16
x = torch.rand(batch_size, 12) # тензор x в программе не менять

# здесь продолжайте программу
# I need more time to evaluate.
model = nn.Sequential()
model.add_module('layer_1', nn.Linear(12, 24, bias=True))
model.add_module('act1', nn.Tanh())
model.add_module('layer_2', nn.Linear(24, 10, bias=True))
model.add_module('act2', nn.Tanh())
model.add_module('out', nn.Linear(10, 1, bias=True))

model.eval()
predict = model(x)
