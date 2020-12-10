import os
import re
import json
import math
import time
import torch
import nvgpu
import psutil
import pathlib
from datetime import timedelta


def format_time(elapsed_time):
    elapsed_time_rounded = int(round(elapsed_time))
    return str(timedelta(seconds=elapsed_time_rounded))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def psnr(outputs, targets):
    mse = torch.nn.MSELoss()(outputs, targets)
    if mse == 0:
        return torch.tensor(float('inf'))
    return torch.tensor(20 * math.log10(255.0 / math.sqrt(mse)))


def get_gpu_memory_usage():
    return sum([info['mem_used'] for info in nvgpu.gpu_info()])


def get_total_gpu_memory():
    return sum([info['mem_total'] for info in nvgpu.gpu_info()])


def get_gpu_memory_usage_percentage():
    return sum([info['mem_used_percent'] for info in nvgpu.gpu_info()])


def get_memory_usage():
    """Return Memory Used in MB"""
    return psutil.virtual_memory().used/1024**2


def get_total_memory():
    """Return Memory Used in MB"""
    return psutil.virtual_memory().total/1024**2


def get_memory_usage_percentage():
    """Return Memory Usage Percentage in MB"""
    return psutil.virtual_memory().percent
