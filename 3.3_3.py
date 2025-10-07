import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

ds = data.TensorDataset(_global_var_data_x, _global_var_target)  # _global_var_data_x - тензор с изображениями цифр размерностью (total, 1, 8, 8) и типом float32; _global_var_data_y - тензор с целочисленными метками классов размерностью (total, ) и типом long (int64).

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),
    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),
    nn.Flatten(),
    nn.Linear(64, 10, bias=True)
)

test_data = data.DataLoader(ds, batch_size=len(ds), shuffle=False)

model.load_state_dict(_global_model_state)

model.eval()
predict = model(test_data.data)
p = torch.argmax(predict, dim=1)
Q = torch.mean((test_data.target == p).float()).item() 

'''
x_test, y_test = next(iter(test_data))
with torch.no_grad():
    p = model(x_test)
    p = torch.argmax(p, dim=1)
    Q = torch.mean(p.flatten() == y_test.flatten()).item()

Q /= len(ds)
'''
