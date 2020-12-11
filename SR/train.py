import os
import re
import sys
import torch
import shutil
import pathlib
import logging
import argparse
import torchvision
import multiprocessing
from torch import nn
from torchvision.transforms.transforms import Scale
from tqdm import tqdm
import torch.optim as optim
from trainer.trainer import Trainer
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataloader.datasets import DIV2K_Square_Dataset
from model.FSRCNN import FSRCNN, FSRCNN_Original
from model.resnet_sr import ResNetPretrainedSS, ResNetPretrainedDS, ResNetPretrainedDSRes
from model.SRCNN import SRCNN
from model.DRRN import DRRN
from model.VDSR import VDSR
from model.UNetSR import UNetSR
import test_all
from logger import get_logger
from utils.loss import loss_sum # input image must be larger than 224x224


def train(mod: nn.Module, criterion, optimizer_, device_, config_, train_dataset, valid_dataset, train_dataloader,
          valid_dataloader, scheduler_lr):
    device_name = torch.cuda.get_device_name(
        device_) if device_.type != 'cpu' else 'cpu'
    print(f"Running on {device_name}\n")
    trainer = Trainer(model=mod,
                      criterion=criterion,
                      optimizer=optimizer_,
                      device=device_,
                      config=config_,
                      train_dataset=train_dataset,
                      valid_dataset=valid_dataset,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      lr_scheduler=scheduler_lr)
    trainer.train()


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1024)
    SR_path = pathlib.Path(__file__).parent.absolute()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###################################################################################################################
    # Below is the configuration you need to set
    ###################################################################################################################
    fsrcnn_original = FSRCNN_Original(
        scale_factor=3, num_channels=3).to(device)

    # SRCNN
    srcnn = SRCNN(in_channel=3).to(device)          # 3 times
    srcnn_150_300 = SRCNN(in_channel=3).to(device)  # 2 times
    srcnn_50_300 = SRCNN(in_channel=3).to(device)   # 6 times
    # FSRCNN
    fsrcnn_custom_3x = FSRCNN(factor=3).to(device)
    fsrcnn_original = FSRCNN_Original(
        scale_factor=3, num_channels=3).to(device)

    # UNetSR
    unetsr = UNetSR().to(device)

    unetsr_config = {
        'epochs': 150,
        'save_period': 5,
        'batch_size': 10,
        'checkpoint_dir': SR_path/'result/UNetSR-100-300-150iter',
        'log_step': 10,
        'start_epoch': 120,
        'criterion': nn.MSELoss(),
        'DATASET_TYPE': 'same_300',
        'low_res': 100,
        'high_res': 300,
        'learning_rate': 0.001,
        'scheduler': None
    }

    srcnn_config = {
        'epochs': 100,
        'save_period': 10,
        'batch_size': 20,
        'checkpoint_dir': SR_path/'result/SRCNN-100-300-100iter-scheduler',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'DATASET_TYPE': 'same_300',
        'low_res': 100,
        'high_res': 300,
        'learning_rate': 0.005,
        'scheduler': {'step_size': 20, 'gamma': 0.4}
    }

    srcnn_config_150_300 = {
        'epochs': 50,
        'save_period': 10,
        'batch_size': 20,
        'checkpoint_dir': SR_path/'result/SRCNN-150-300-100iter-scheduler',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'DATASET_TYPE': 'same_300',
        'low_res': 150,
        'high_res': 300,
        'learning_rate': 0.005,
        'scheduler': {'step_size': 20, 'gamma': 0.4}
    }

    srcnn_config_50_300 = {
        'epochs': 50,
        'save_period': 10,
        'batch_size': 20,
        'checkpoint_dir': SR_path/'result/SRCNN-50-300-100iter-scheduler',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'DATASET_TYPE': 'same_300',
        'low_res': 50,
        'high_res': 300,
        'learning_rate': 0.005,
        'scheduler': {'step_size': 20, 'gamma': 0.4}
    }

    resnet_pretrained_ss = ResNetPretrainedSS()
    resnet_pretrained_ss_config = {
        'epochs': 1,
        'save_period': 10,
        'batch_size': 20,
        'checkpoint_dir': SR_path/'result/resnet_pretrained_ss_50',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'DATASET_TYPE': 'same',
        'low_res': 200,
        'high_res': 600,
        'learning_rate': 0.005,
        'scheduler': {'step_size': 10, 'gamma': 0.4}
    }

    # models = [srcnn, srcnn_150_300, srcnn_50_300]
    # configs = [srcnn_config,
    #            srcnn_config_150_300, srcnn_config_50_300]
    models = [resnet_pretrained_ss, resnet_pretrained_ss]
    configs = [resnet_pretrained_ss_config, {
        'epochs': 1,
        'save_period': 10,
        'batch_size': 10,
        'checkpoint_dir': SR_path/'result/testlog',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'DATASET_TYPE': 'same',
        'low_res': 200,
        'high_res': 600,
        'learning_rate': 0.005,
        'scheduler': {'step_size': 10, 'gamma': 0.4}
    }]
    ###################################################################################################################
    # Above is the configuration you need to set
    ###################################################################################################################

    DIV2K_path = (SR_path.parent/'datasets'/'DIV2K').absolute()
    DS = DIV2K_Square_Dataset

    train_set_percentage = 0.96

    for i in range(len(models)):
        torch.cuda.empty_cache()
        model = models[i]
        config = configs[i]
        ######################### setup workspace #################################################################
        if config['start_epoch'] <= 1 and config["checkpoint_dir"].exists():
            # not training from a checkpoint
            shutil.rmtree(config["checkpoint_dir"])
        ######################### setup logger ####################################################################
        config["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
        # logging.basicConfig(format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
        logger = get_logger(os.path.basename(pathlib.Path(
            config["checkpoint_dir"]).absolute()), config["checkpoint_dir"]/'log.log')
        # logging.basicConfig(
        #     filename=f'{config["checkpoint_dir"]/"log.log"}', level=logging.INFO,
        #     format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
        # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        ######################### setup logger ####################################################################
        DATASET_TYPE = config['DATASET_TYPE']
        lr_number, hr_number = config['low_res'], config['high_res']
        train_in_dir, train_label_dir = DIV2K_path / DATASET_TYPE / \
            f'train_{lr_number}', DIV2K_path / \
            DATASET_TYPE / f'train_{hr_number}'
        test_in_dir, test_label_dir = DIV2K_path / DATASET_TYPE / \
            f'valid_{lr_number}', DIV2K_path / \
            DATASET_TYPE / f'valid_{hr_number}'
        # log training dataset info
        logger.info("=" * 100)
        logger.info(f"training model: {model.__class__}")
        logger.info(f"start learning rate: {config['learning_rate']}")
        logger.info(f"number of epochs: {config['epochs']}")
        logger.info(f"batch_size: {config['batch_size']}")
        logger.info(f"log step: {config['log_step']}")
        logger.info(f"criterion: {config['criterion']}")
        logger.info(f"dataset_type: {DATASET_TYPE}")
        logger.info(f"low resolution: {lr_number}")
        logger.info(f"high resolution: {hr_number}")
        logger.info(f"checkpoint_dir: {config['checkpoint_dir']}")
        logger.info(f"scheduler: {config['scheduler']}")
        logger.info("=" * 100)

        # optimizer = optim.SGD(model.parameters(), lr=0.05)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        if 'scheduler' in config and config['scheduler'] is not None:
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=config['scheduler']['step_size'], gamma=config['scheduler']['gamma'])
        else:
            scheduler = None
        dataset = DS(input_dir=train_in_dir, target_dir=train_label_dir,
                     transform=transforms.ToTensor())
        num_train = int(train_set_percentage * len(dataset))
        num_valid = int(len(dataset) - num_train)
        train_set, valid_set = torch.utils.data.random_split(
            dataset, [num_train, num_valid])
        train_loader = DataLoader(
            dataset=train_set, batch_size=config['batch_size'],
            shuffle=True, num_workers=multiprocessing.cpu_count(),
            pin_memory=True)
        valid_loader = DataLoader(
            dataset=valid_set, batch_size=config['batch_size'],
            shuffle=True, num_workers=multiprocessing.cpu_count(),
            pin_memory=True)
        try:
            train(model,
                  config['criterion'],
                  optimizer,
                  device,
                  config,
                  train_set,
                  valid_set,
                  train_loader,
                  valid_loader,
                  scheduler)
        except Exception as e:
            logger.error(e)
            logger.error("Failed! Skip to next model if there is any left.")
            continue

        # run test images
        try:
            weight_path = pathlib.Path(config['checkpoint_dir'])/'weights'
            weight_files = sorted(os.listdir(weight_path), key=lambda filename: int(
                re.findall('epoch(\d{1,})\.pth', filename)[0]))
            if len(weight_files) != 0:
                logger.info("Running tests")
                test_all.main(
                    config['DATASET_TYPE'],
                    config['low_res'],
                    config['high_res'],
                    weight_path / weight_files[-1],
                    config['checkpoint_dir'] / 'test', model.__class__.__name__)
        except KeyError as e:
            logger.error(
                "Test images failed to generate. Likely your model is not registered in the test_all.py file. Modify the map dictionary called model_map")