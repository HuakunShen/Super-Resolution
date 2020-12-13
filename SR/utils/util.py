import os
import re
import json
import math
import time
import torch
import nvgpu
import shutil
import psutil
import pathlib
from datetime import timedelta
from torchvision.transforms import ToTensor


def format_time(elapsed_time):
    """format time"""
    elapsed_time_rounded = int(round(elapsed_time))
    return str(timedelta(seconds=elapsed_time_rounded))


def get_lr(optimizer):
    """Get learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def psnr_PIL(output, target):
    """output and target are both PIL Image"""
    output_tensor = ToTensor()(output)
    target_tensor = ToTensor()(target)
    return psnr(output_tensor, target_tensor)


def psnr(outputs, targets):
    mse = torch.nn.MSELoss()(outputs, targets)
    if mse == 0:
        return torch.tensor(float('inf'))
    return torch.tensor(20 * math.log10(255.0 / math.sqrt(mse)))


def get_divider_str(msg: str, length: int = 100):
    result = ""
    space_left = length - 2 - len(msg)
    if space_left <= 0:
        return msg
    elif space_left % 2 == 0:
        left, right = space_left // 2, space_left // 2
    else:
        left, right = space_left // 2, space_left // 2 + 1
    return f"{left*'='} {msg} {right*'='}"


def save_checkpoint_state(state, epoch: int, is_best, checkpoint_dir: pathlib.Path):
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    f_path = checkpoint_dir / f'epoch{epoch}.pth'
    torch.save(state, f_path)
    if is_best:
        best_fpath = checkpoint_dir / 'best.pth'
        shutil.copyfile(f_path, best_fpath)


def load_checkpoint_state(checkpoint_fpath):
    return torch.load(checkpoint_fpath)
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # return model, optimizer, checkpoint['epoch']


if __name__ == "__main__":
    msg = 'configuration parameterss'
    print(get_divider_str(msg, 100))
    print(len(get_divider_str(msg, 100)))
