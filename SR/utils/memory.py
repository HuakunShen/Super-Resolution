import subprocess as sp
import numpy as np
import nvgpu
import psutil


def get_gpu_memory_usage(mode=0):
    """There may be error on Windows, this method doesn't do error checking"""
    if mode == 0:
        return np.nan
    elif mode == 1:
        return sum([info['mem_used'] for info in nvgpu.gpu_info()])
    elif mode == 2:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        memory_used_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
        return np.sum(memory_used_values)
    else:
        raise ValueError("invalid mode, mode = enum(0,1,2)")


def get_total_gpu_memory(mode=0):
    """There may be error on Windows, this method doesn't do error checking"""
    if mode == 0:
        return np.nan
    elif mode == 1:
        return sum([info['mem_total'] for info in nvgpu.gpu_info()])
    elif mode == 2:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_total_values = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
        return np.sum(memory_total_values)
    else:
        raise ValueError("invalid mode, mode = enum(0,1,2)")


def get_gpu_memory_usage_percentage(mode=0):
    """There may be error on Windows, this method doesn't do error checking"""
    if mode == 0:
        return np.nan
    elif mode == 1:
        return sum([info['mem_used_percent'] for info in nvgpu.gpu_info()])
    elif mode == 2:
        return float(get_gpu_memory_usage(mode=2)) / float(get_total_gpu_memory(mode=2))
    else:
        raise ValueError("invalid mode, mode = enum(0,1,2)")


def get_memory_usage():
    """Return Memory Used in MB"""
    return psutil.virtual_memory().used / 1024 ** 2


def get_total_memory():
    """Return Memory Used in MB"""
    return psutil.virtual_memory().total / 1024 ** 2


def get_memory_usage_percentage():
    """Return Memory Usage Percentage in MB"""
    return psutil.virtual_memory().percent