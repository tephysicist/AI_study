import torch

t_rnd = torch.randint(-3, 5, (100, ), dtype=torch.float32) # значения этого тензора в программе не менять
t_mean = torch.mean(t_rnd)