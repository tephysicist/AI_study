import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


# the class is from 4.3_4.py
class CharsDataset(data.Dataset):
    def __init__(self, prev_chars):
        self.prev_chars = prev_chars
        self.lines = _global_var_text
        
        self.alphabet = set("".join(self.lines).lower())
        self.int_to_alpha = dict(enumerate(sorted(self.alphabet)))
        self.alpha_to_int = {b: a for a, b in self.int_to_alpha.items()}
        self.num_characters = len(self.alphabet) # size of the alphabet
        self.onehots = torch.eye(self.num_characters)

        data = []
        targets = []
        for line in self.lines:
            line = line.lower()
            for i in range(len(line) - self.prev_chars):
                data.append([self.alpha_to_int[line[x]] for x in range(i, i+self.prev_chars)])
                ch = line[i+self.prev_chars]
                targets.append(self.alpha_to_int[ch])
        
        self.data = torch.tensor(data)
        self.targets = torch.tensor(targets)
        self.length = len(data)
        
    def __getitem__(self, item):
        return self.onehots[self.data[item]], self.targets[item]
        
    def __len__(self):
        return self.length


class RNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 32):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.rnn(x)
        y = self.out(h)
        return y



d_train = CharsDataset(10)
train_data = data.DataLoader(d_train, batch_size = 8, shuffle=True, drop_last=False)


model = RNeuralNetwork(d_train.num_characters, d_train.num_characters)


optimizer = optim.Adam(params=model.parameters(), lr=0.01) 
loss_func = nn.CrossEntropyLoss()


epochs = 1
model.train()


for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train).squeeze(0) 
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


model.eval()
predict = "нейронная сеть ".lower()
total = 20


for _ in range(total):
    _data = d_train.onehots[[d_train.alpha_to_int[predict[-x]] for x in range(d_train.prev_chars, 0, -1)]]
    with torch.no_grad():
        p = model(_data.unsqueeze(0)).squeeze(0)
    indx = torch.argmax(p, dim=1)
    predict += d_train.int_to_alpha[indx.item()]

print(predict)
