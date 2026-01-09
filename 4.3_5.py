import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim



# сюда копируйте класс CharsDataset из предыдущего подвига
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


# здесь объявляйте класс модели нейронной сети
class RNNNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size = 32, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=1, hidden_size=self.hidden_size, num_layers=1, nonlinearity='tanh', bias=False, batch_first=True)
        self.layer = nn.Linear(self.hidden_size, output_size, bias=True)



# сюда копируйте объекты d_train и train_data
d_train = CharsDataset(10)
train_data = data.DataLoader(d_train, batch_size = 8, shuffle=True, drop_last=False)


model = # создайте объект модели


optimizer = # оптимизатор Adam с шагом обучения 0.01
loss_func = # функция потерь - CrossEntropyLoss


epochs = 1 # число эпох (это конечно, очень мало, в реальности нужно от 100 и более)
# переведите модель в режим обучения


for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = # вычислите прогноз модели для x_train
        loss = # вычислите потери для predict и y_train


        # выполните один шаг обучения (градиентного спуска)


# переведите модель в режим эксплуатации
predict = "нейронная сеть ".lower() # начальная фраза
total = 20 # число прогнозируемых символов (дополнительно к начальной фразе)


# выполните прогноз следующих total символов
# выведите полученную строку на экран
