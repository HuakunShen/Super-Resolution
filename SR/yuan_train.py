import torch
from torch import nn
import multiprocessing
import torch.optim as optim
from model.SRCNN import SRCNN
from trainer import train
from config import SR_PATH
from utils.loss import LossSum, FeatureExtractor


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ###################################################################################################################
    # Below is the configuration you need to set
    ###################################################################################################################
    fe = FeatureExtractor().to(device)
    criterion = LossSum(fe)
    criterion(1, 2)

    srcnn = SRCNN(in_channel=3).to(device)
    srcnn_config = {
        'epochs': 1,
        'save_period': 10,
        'batch_size': 20,
        'checkpoint_dir': SR_PATH / 'result/SRCNN-100-300-scheduler',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'dataset_type': 'same_300',
        'low_res': 100,
        'high_res': 300,
        'device': device,
        'scheduler': {'step_size': 1, 'gamma': 0.4},
        'optimizer': optim.Adam(srcnn.parameters(), lr=0.001),
        'train_set_percentage': 0.96,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1
    }
    models = [srcnn]
    configs = [srcnn_config]
    ###################################################################################################################
    # Above is the configuration you need to set
    ###################################################################################################################

    train.run(models, configs)
