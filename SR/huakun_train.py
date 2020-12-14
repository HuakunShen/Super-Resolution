import torch
from torch import nn
import multiprocessing
import torch.optim as optim
from model.SRCNN import SRCNN
from trainer import train
from config import SR_PATH
from model.UNetSR import UNetNoTop, UNetD4, UNetSR
from model.SRCNN import SRCNN
from utils.loss import FeatureExtractor, LossSum
import pathlib


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RESULT_PATH = pathlib.Path('/media/hacker/PCshared/Super-Resolution')
    ###################################################################################################################
    # Below is the configuration you need to set
    ###################################################################################################################
    fe = FeatureExtractor().to(device)
    criterion = LossSum(fe)
    # srcnn = SRCNN().to(device)
    # srcnn_config = {
    #     'epochs': 50,
    #     'save_period': 10,
    #     'batch_size': 10,
    #     'checkpoint_dir': RESULT_PATH / 'result/srcnn_100_300_perceptual_loss_2',
    #     'log_step': 10,
    #     'start_epoch': 1,
    #     'criterion': criterion,
    #     'dataset_type': 'same_300',
    #     'low_res': 100,
    #     'high_res': 300,
    #     'device': device,
    #     'scheduler': {
    #         'step_size': 8,
    #         'gamma': 0.6
    #     },
    #     'optimizer': optim.Adam(srcnn.parameters(), lr=0.003),
    #     'train_set_percentage': 0.9,
    #     'num_worker': multiprocessing.cpu_count(),
    #     'test_all_multiprocess_cpu': 1,
    #     'test_only': False
    # }

    unetsr = UNetSR(in_c=3, out_c=3, output_paddings=[1, 1]).to(device)
    unet_config = {
        'epochs': 150,
        'save_period': 10,
        'batch_size': 8,
        'checkpoint_dir': RESULT_PATH / 'result/unetsr_100_300_perceptual_loss_w_seed',
        'log_step': 10,
        'start_epoch': 100,
        'criterion': criterion,
        'dataset_type': 'same_300',
        'low_res': 100,
        'high_res': 300,
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.85
        },
        'optimizer': optim.Adam(unetsr.parameters(), lr=0.002),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }
    models = [unetsr]
    configs = [unet_config]
    # models = [unetsr]
    # configs = [unet_config]

    ###################################################################################################################
    # Above is the configuration you need to set
    ###################################################################################################################

    train.run(models, configs)
