import os
import json
import math
import time
import torch
from datetime import timedelta


def format_time(elapsed_time):
    elapsed_time_rounded = int(round(elapsed_time))
    return str(timedelta(seconds=elapsed_time_rounded))


if __name__ == '__main__':
    t_0 = time.time()
    time.sleep(2)
    print(format_time(time.time() - t_0))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def psnr(outputs, targets):
    mse = torch.nn.MSELoss()(outputs, targets)
    if mse == 0:
        return torch.tensor(float('inf'))
    return torch.tensor(20 * math.log10(255.0 / math.sqrt(mse)))
