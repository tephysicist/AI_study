import torch
import torch.nn as nn
import torch.utils.data as data


# здесь объявляйте класс CharsDataset
class CharsDataset(data.Dataset):
    def __init__(self, prev_chars):
        self.prev_chars = prev_chars
        self.lines = _global_var_text
        
        self.alphabet = set("".join(self.lines).lower())
        self.int_to_alpha = dict(enumerate(sorted(self.alphabet)))
        self.alpha_to_int = {b: a for a, b in self.int_to_alpha.items()}
        self.num_characters = len(self.alphabet)
        self.onehots = torch.eye(self.num_characters)
        
    def __getitem__(self, item):
        _data = torch.vstack([self.onehots[self.alpha_to_int[x]] for x in range()])




# здесь продолжайте программу
