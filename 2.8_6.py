import torch
import torch.nn as nn
import torch.utils.data as data


class FuncDataset(data.Dataset): # имя класса MyDataset может быть любым
	def __init__(self): # инициализация переменных объекта класса
    	self.coord_x = torch.arange(-5, 5, 0.1)
        self.coord_y = 2*torch.exp(-x/2) + 0.2*torch.sin(x/10) - 5
        self.length = len(self.coord_x)
 
	def __getitem__(self, item): # возвращение образа выборки по индексу item
		return self.coord_x[item], self.coord_y[item]
 
	def __len__(self): # возвращение размера выборки
    	return self.length


d_train = FuncDataset()


x13, y13 = d_train[13]
total = len(d_train)
