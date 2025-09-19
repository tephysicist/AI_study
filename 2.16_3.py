import torch
import torch.nn as nn

batch_size=16
x = torch.rand(batch_size, 12) # тензор x в программе не менять

# здесь продолжайте программу
# здесь продолжайте программу
model = nn.Sequential(
	nn.Linear(12, 24, bias=True),
	nn.Tanh(),
    nn.Linear(24, 10, bias=True),
    nn.Linear(10, 1, bias=True)
)
