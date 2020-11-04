import math
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def print_options(opt, sort=False):
    keys = list(vars(opt).keys())
    if sort:
        keys.sort()
    for arg in keys:
        print(arg, ":", getattr(opt, arg))


class CosineAnnealingCustom:

    def __init__(self, begin, end, t_max):
        self.T_max = t_max
        self.begin = begin
        self.end = end
        self.inv = begin < end

    def get(self, epoch):
        if not self.inv:
            return self.end + (self.begin - self.end) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        else:
            return self.begin + (self.end - self.begin) * (1 - math.cos(math.pi * epoch / self.T_max)) / 2


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
