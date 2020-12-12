import logging
from typing import Union
import subprocess as sp
import numpy as np
from utils.memory import get_gpu_memory_usage, get_total_gpu_memory, get_memory_usage, get_total_memory


class MemoryProfiler(object):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.gpu_mem_mode = 0  # mode 0: no gpu info available, mode 1: nvgpu, mode 2: subprocess nvidia-smi
        try:
            self.max_gpu_memory_used = get_gpu_memory_usage(mode=1)
            self.gpu_mem_mode = 1
        except IndexError as e:
            self.logger.error(e)
            self.logger.error("Got an IndexError exception while getting gpu memory with nvgpu, try using nvidia-smi"
                              " query with subprocess ")
            try:
                self.max_gpu_memory_used = get_gpu_memory_usage(mode=2)
                self.gpu_mem_mode = 2
            except sp.CalledProcessError:
                self.logger.error(e)
                self.logger.error(
                    "Got subprocess.CalledProcessError when calling nvidia-smi. Likely not installed in OS. GPU Memory "
                    "will not be available for MemoryProfiler")
                self.max_gpu_memory_used = np.nan
        self.max_mem_used = get_memory_usage()

    def update_n_log(self, epoch: Union[int, None]):
        self.update()
        self.log(epoch)

    def log(self, epoch: Union[int, None]):
        if epoch is not None:
            if self.gpu_mem_mode != 0:
                self.logger.debug(f'MAX GPU Usage (epoch {epoch}): {self.max_gpu_memory_used}MB')
            self.logger.debug(f"MAX Memory Usage (epoch {epoch}): {round(self.max_mem_used, 2)}MB")

    def update(self):
        if self.gpu_mem_mode != 0:
            self.max_gpu_memory_used = max(self.max_gpu_memory_used, get_gpu_memory_usage(mode=self.gpu_mem_mode))
        self.max_mem_used = max(self.max_mem_used, get_memory_usage())

    def log_final_message(self):
        self.logger.info(
            f"Max GPU Usage: {self.max_gpu_memory_used}MB/{get_total_gpu_memory(mode=self.gpu_mem_mode)}MB "
            f"({round(self.max_gpu_memory_used / get_total_gpu_memory(mode=self.gpu_mem_mode) * 100, 2)}%)")
        self.logger.info(
            f"Max Memory Usage: {self.max_mem_used}MB/{get_total_memory()}MB "
            f"({round(self.max_mem_used / get_total_memory() * 100, 2)}%)")
