import torch
from torch import nn
import multiprocessing
import torch.optim as optim
from model.SRCNN import SRCNN
from trainer import train
from config import SR_PATH
from model.UNetSR import UNetNoTop, UNetD4
from model.SRCNN import SRCNN

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ###################################################################################################################
    # Below is the configuration you need to set
    ###################################################################################################################
    srcnn = SRCNN().to(device)
    srcnn_config = {
        'epochs': 30,
        'save_period': 5,
        'batch_size': 20,
        'checkpoint_dir': SR_PATH / 'result/srcnn_new_framework',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'dataset_type': 'same_300',
        'low_res': 100,
        'high_res': 300,
        # 'learning_rate': 0.003,
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.7
        },
        'optimizer': optim.Adam(srcnn.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1
    }

    unetNoTop = UNetNoTop().to(device)
    unetD4 = UNetD4().to(device)
    unetNoTop_config = {
        'epochs': 100,
        'save_period': 20,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/unetNoTop',
        'log_step': 10,
        'start_epoch': 10,
        'criterion': nn.MSELoss(),
        'dataset_type': 'same_300',
        'low_res': 50,
        'high_res': 200,
        # 'learning_rate': 0.003,
        'device': device,
        'scheduler': {
            'step_size': 10,
            'gamma': 0.5
        },
        'optimizer': optim.Adam(unetNoTop.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1
    }
    unetD4_config = {
        'epochs': 100,
        'save_period': 20,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/unetD4',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'dataset_type': 'same_300',
        'low_res': 50,
        'high_res': 200,
        'learning_rate': 0.003,
        'device': device,
        'scheduler': {
            'step_size': 10,
            'gamma': 0.5
        },
        'optimizer': optim.Adam(unetNoTop.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1
    }
    # models = [srcnn, unetNoTop, unetD4]
    # configs = [srcnn_config, unetNoTop_config, unetD4_config]
    models = [srcnn]
    configs = [srcnn_config]

    ###################################################################################################################
    # Above is the configuration you need to set
    ###################################################################################################################

    train.run(models, configs)
