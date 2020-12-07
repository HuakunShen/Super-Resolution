import os
import json
import time
from datetime import timedelta


def format_time(elapsed_time):
    elapsed_time_rounded = int(round(elapsed_time))
    return str(timedelta(seconds=elapsed_time_rounded))


if __name__ == '__main__':
    t_0 = time.time()
    time.sleep(2)
    print(format_time(time.time() - t_0))