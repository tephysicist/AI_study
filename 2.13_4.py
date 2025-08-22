import torch
import torch.utils.data as data

class FuncDataset(data.Dataset):
    def __init__(self):
        _x = torch.arange(-4, 4, 0.01)
        self.data = _x
        self.target = _x**2 + 0.5*_x - torch.sin(5*_x)
        self.length = len(self.data) # размер обучающей выборки

    def __getitem__(self, item):
        return self.data[item], self.target[item] # возврат образа по индексу item в виде кортежа: (данные, целевое значение)

    def __len__(self):
        return self.length # возврат размера выборки

dataset = FuncDataset()
d_train, d_val = data.random_split(dataset, [0.8, 0.2])
train_data = data.DataLoader(d_train, batch_size = 16, shuffle=True, drop_last=False)
train_data_val = data.DataLoader(d_train, batch_size = 100, shuffle=False, drop_last=False)
