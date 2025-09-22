import torch
import torch.nn as nn

batch_size=12
x = torch.rand(batch_size, 64) # тензор x в программе не менять

# здесь продолжайте программу
block_bm_dp = nn.Sequential(
    nn.Linear(32, 32, bias=False),
    nn.ELU(),
    nn.BatchNorm1d(32),
    nn.Dropout1d(0.3)
)

model = nn.Sequential()
model.add_module('input', nn.Linear(64, 32, bias=True))
model.add_module('act1', nn.ReLU())
model.add_module('block1', block_bm_dp)
model.add_module('block2', block_bm_dp)
model.add_module('block3', block_bm_dp)
model.add_module('output', nn.Linear(32, 10, bias=True))

predict = model(x)
