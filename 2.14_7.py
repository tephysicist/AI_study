import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ImageNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=64, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=10)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

model = ImageNN()
ds = data.TensorDataset(_global_var_data_x, _global_var_target)
d_train, d_test = data.random_split(ds, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=16, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.1)
loss_func = nn.CrossEntropyLoss()
