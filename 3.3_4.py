import torch
import torch.nn as nn

# здесь объявляйте класс ImageNormalize
class ImageNormalize(nn.Module):        
    def forward(self, x):
        min, max = x.min(), x.max()
        x = (x - min)/(max - min)
        return x 

# генерация образов выборки
total = 100 # размер выборки
H, W = 32, 32 # размер изображений
circle = torch.tensor([[0, 0, 0, 255, 255, 255, 255, 0, 0, 0],
                       [0, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                       [0, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                       [0, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                       [0, 255, 255, 255, 255, 255, 255, 255, 255, 0],
                       [0, 0, 0, 255, 255, 255, 255, 0, 0, 0]], dtype=torch.float32)
Hc, Wc = circle.size()




def _generate_img(_H, _W, _Hc, _Wc, _x, _y, _circle, _tr): # вспомогательная функция
    img = torch.rand(_H, _W) * 20
    img[_x:_x+_Hc, _y:_y+Wc] = _circle
    return _tr(img.view(1, 1, _H, _W))



transform = ImageNormalize()  # создайте объект класса ImageNormalize
data_y = torch.tensor([(torch.randint(0, H-Hc, (1, )), torch.randint(0, W-Wc, (1, ))) for _ in range(total)])
data_x = torch.cat([_generate_img(H, W, Hc, Wc, _x[0], _x[1], circle, transform) for _x in data_y], dim=0)


torch.manual_seed(1)
# здесь создавайте модель (обязательно сразу после команды torch.manual_seed(1))
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, return_indices=False, ceil_mode=False),
    nn.Flatten(),
    nn.Linear(2048, 2, bias=True) #2048
)

model.eval()
loss_func = nn.MSELoss()
with torch.no_grad():
    predict = model(data_x) # пропустите через модель выборку data_x
    Q = loss_func(predict, data_y.float()) # вычислите величину потерь, используя функцию loss_func = nn.MSELoss()
