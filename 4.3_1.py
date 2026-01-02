import torch
import torch.nn as nn


sigma = 0.1 # стандартное отклонение отсчетов последовательности
r = 0.9 # коэффициент регрессии
sigma_noise = sigma * (1 - r * r) ** 0.5 # стандартное отклонение случайных величин


total = 100 # длина генерируемой последовательности
noise = torch.randn((total, )) # случайные величины, подаваемые на вход модели
x0 = torch.randn((1, )) * sigma # начальное значение вектора скрытого состояния


# здесь продолжайте программу
model = nn.Sequential(
    nn.RNN(input_size=in_features, hidden_size=15, num_layers=1, nonlinearity='tanh', bias=True, batch_first=True)
)



model.eval()
x = model(x)
