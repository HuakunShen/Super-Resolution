import os
import re
import json
import math
import time
import torch
import pathlib
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


def get_loss_plot_names(checkpoint_path):
    checkpoint_path = pathlib.Path(checkpoint_path).absolute()
    if not checkpoint_path.exists():
        return checkpoint_path/'train_loss.png', checkpoint_path/'valid_loss.png'
    files = os.listdir(checkpoint_path)
    first_gen_loss_files = list(
        filter(lambda x: re.match('.+_loss\.png', x), files))
    extra_gen_loss_files = list(
        filter(lambda x: re.match('.+_loss_\d+\.png', x), files))
    if len(first_gen_loss_files) == 0 and len(extra_gen_loss_files) == 0:
        return checkpoint_path/'train_loss.png', checkpoint_path/'valid_loss.png'
    elif len(first_gen_loss_files) != 0 and len(extra_gen_loss_files) == 0:
        return checkpoint_path/'train_loss_1.png', checkpoint_path/'valid_loss_1.png'
    else:
        extra_gen_numbers = [int(re.findall('.+_loss_(\d+)\.png', file)[0])
                             for file in extra_gen_loss_files]
        max_extra_gen = max(extra_gen_numbers)
        return checkpoint_path/f'train_loss_{max_extra_gen+1}.png', checkpoint_path/f'valid_loss_{max_extra_gen+1}.png'
