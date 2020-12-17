import torch
from torch import nn
import multiprocessing
import torch.optim as optim
from model.SRCNN import SRCNN
from trainer import train
from config import SR_PATH, TEXT_DATASET_PATH, DIV2K_DATASET_PATH
from model.UNetSR import UNetNoTop, UNetD4, UNetSR
from model.SRCNN import SRCNN
from model.resnet_sr import ResNetPretrainedDS
from utils.loss import FeatureExtractor, LossSum
import pathlib


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RESULT_PATH = pathlib.Path(
        '~/Documents/Super-Resolution/result')
    ###################################################################################################################
    # Below is the configuration you need to set
    ###################################################################################################################
    fe = FeatureExtractor().to(device)
    criterion = LossSum(fe)
    unetsr = UNetSR(in_c=3, out_c=3, output_paddings=[1, 1]).to(device)
    unet_config = {
        'epochs': 80,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': RESULT_PATH / 'unetd4-blur-5-perceptual',
        'start_epoch': 1,
        'criterion': criterion,
        'dataset': TEXT_DATASET_PATH,
        'dataset_type': 'same',
        'low_res': 'BlurRadius5',
        'high_res': 'Target300x300',
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.91
        },
        # 'scheduler': None,
        'optimizer': optim.Adam(unetsr.parameters(), lr=0.002),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }

    models = [unetsr]
    configs = [unet_config]

    ###################################################################################################################
    # Above is the configuration you need to set
    ###################################################################################################################

    train.run(models, configs)
