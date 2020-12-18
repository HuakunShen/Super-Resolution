import torch
from torch import nn
import multiprocessing
import torch.optim as optim

from model.DRRN import DRRN
from model.VDSR import VDSR
from utils.loss import FeatureExtractor, LossSum
from trainer import train
from config import *

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ###################################################################################################################
    # Below is the configuration you need to set
    ###################################################################################################################
    drrn_img_percep = DRRN().to(device)
    drrn_img_percep_config = {
        'epochs': 20,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/IMAGE/DRRN-PERCEP-1-100-scheduler',
        'log_step': 5,
        'start_epoch': 1,
        'criterion': LossSum(FeatureExtractor().to(device)),
        'dataset': DIV2K_DATASET_PATH,
        'dataset_type': 'same_300',
        'low_res': '100',
        'high_res': '300',
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.8
        },
        # 'scheduler': None,
        'optimizer': optim.Adam(drrn_img_percep.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }
    drrn_img_mle = DRRN().to(device)
    drrn_img_mle_config = {
        'epochs': 20,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/IMAGE/DRRN-MLE-1-100-scheduler',
        'log_step': 5,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'dataset': DIV2K_DATASET_PATH,
        'dataset_type': 'same_300',
        'low_res': '100',
        'high_res': '300',
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.8
        },
        # 'scheduler': None,
        'optimizer': optim.Adam(drrn_img_mle.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }
    drrn_text_percep = DRRN().to(device)
    # unetd4 = UNetD4(in_c=3, out_c=3).to(device)
    drrn_text_percep_config = {
        'epochs': 20,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/TEXT/DRRN-PERCEP-1-100-scheduler',
        'log_step': 5,
        'start_epoch': 1,
        'criterion': LossSum(FeatureExtractor().to(device)),
        'dataset': TEXT_DATASET_PATH,
        'dataset_type': 'same',
        'low_res': 'BlurRadius3',
        'high_res': 'Target300x300',
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.8
        },
        # 'scheduler': None,
        'optimizer': optim.Adam(drrn_text_percep.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }
    drrn_text_mle = DRRN().to(device)
    # unetd4 = UNetD4(in_c=3, out_c=3).to(device)
    drrn_text_mle_config = {
        'epochs': 20,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/TEXT/DRRN-MLE-1-100-scheduler',
        'log_step': 5,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'dataset': TEXT_DATASET_PATH,
        'dataset_type': 'same',
        # 'low_res': 'Resize50x50',
        # 'high_res': 'Target300x300',
        'low_res': 'BlurRadius3',
        'high_res': 'Target300x300',
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.8
        },
        # 'scheduler': None,
        'optimizer': optim.Adam(drrn_text_mle.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }
    vdsr_img_percep = VDSR().to(device)
    vdsr_img_percep_config = {
        'epochs': 20,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/IMAGE/VDSR-PERCEP-1-100-scheduler',
        'log_step': 5,
        'start_epoch': 1,
        'criterion': LossSum(FeatureExtractor().to(device)),
        'dataset': DIV2K_DATASET_PATH,
        'dataset_type': 'same_300',
        'low_res': '100',
        'high_res': '300',
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.8
        },
        # 'scheduler': None,
        'optimizer': optim.Adam(vdsr_img_percep.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }
    vdsr_img_mle = VDSR().to(device)
    vdsr_img_mle_config = {
        'epochs': 20,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/IMAGE/VDSR-MLE-1-100-scheduler',
        'log_step': 5,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'dataset': DIV2K_DATASET_PATH,
        'dataset_type': 'same_300',
        'low_res': '100',
        'high_res': '300',
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.8
        },
        # 'scheduler': None,
        'optimizer': optim.Adam(vdsr_img_mle.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }
    vdsr_text_percep = VDSR().to(device)
    # unetd4 = UNetD4(in_c=3, out_c=3).to(device)
    vdsr_text_percep_config = {
        'epochs': 20,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/TEXT/VDSR-PERCEP-1-100-scheduler',
        'log_step': 5,
        'start_epoch': 1,
        'criterion': LossSum(FeatureExtractor().to(device)),
        'dataset': TEXT_DATASET_PATH,
        'dataset_type': 'same',
        # 'low_res': 'Resize50x50',
        # 'high_res': 'Target300x300',
        'low_res': 'BlurRadius3',
        'high_res': 'Target300x300',
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.8
        },
        # 'scheduler': None,
        'optimizer': optim.Adam(vdsr_text_percep.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }
    vdsr_text_mle = VDSR().to(device)
    # unetd4 = UNetD4(in_c=3, out_c=3).to(device)
    vdsr_text_mle_config = {
        'epochs': 20,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': SR_PATH / 'result/TEXT/VDSR-MLE-1-100-scheduler',
        'log_step': 5,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'dataset': TEXT_DATASET_PATH,
        'dataset_type': 'same',
        'low_res': 'BlurRadius3',
        'high_res': 'Target300x300',
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.8
        },
        # 'scheduler': None,
        'optimizer': optim.Adam(vdsr_text_mle.parameters(), lr=0.005),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }
    models = [drrn_img_percep, drrn_text_percep, vdsr_img_percep, vdsr_text_percep]
    configs = [drrn_img_percep_config, drrn_text_percep_config, vdsr_img_percep_config, vdsr_text_percep_config]
    ###################################################################################################################
    # Above is the configuration you need to set
    ###################################################################################################################

    train.run(models, configs)
