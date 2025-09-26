import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(64, 32, bias=True),
            nn.ReLU()
        )
        self.blocks = nn.ModuleDict({
            'block_1' : nn.Sequential(nn.Linear(32, 32, bias=False), nn.ELU(), nn.BatchNorm1d(32)),
            'block_2' : nn.Sequential(nn.Linear(32, 32, bias=True), nn.ReLU(), nn.Dropout1d(0.4))
        })
        self.output = nn.Sequential(
            nn.Linear(32, 10, bias=True)
        )
    def forward(self, x, type_block='block_1'):
        x = self.input(x)
        x = self.blocks[type_block](x) if type_block in self.blocks else self.blocks['block_1']
        return self.output(x)


batch_size = 100
x = torch.rand(batch_size, 64) # тензор x в программе не менять

model = NN()
model.eval()
predict = model(x, 'block_2')
