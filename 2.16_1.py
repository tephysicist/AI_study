import torch
import torch.nn as nn

batch_size=8
x = torch.rand(batch_size, 5) # тензор x в программе не менять

# здесь продолжайте программу
model = nn.Sequential(
	nn.Linear(5, 16, bias=False),
	nn.ReLU(),
    nn.BatchNorm1d(16),
	nn.Linear(16, 3)
)

model.eval()
predict = model(x)
