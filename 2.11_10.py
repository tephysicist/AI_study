import torch
t_rnd = torch.randint(-3, 5, (100, ), dtype=torch.float32) # значения этого тензора в программе не менять
t_mean = torch.mean(t_rnd).item()
t_max = t_rnd[:5].max().item()
t_min = t_rnd[-3:].min().item()
