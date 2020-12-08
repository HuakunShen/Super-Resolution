import os
import re
import torch
import pathlib
import logging
import argparse
import torchvision
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
from model.DRRN import DRRN
from model.VDSR import VDSR
import test_all


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
    SR_path = pathlib.Path(__file__).parent.parent.absolute()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fsrcnn_original = FSRCNN_Original(
        scale_factor=3, num_channels=3).to(device)
    fsrcnn_custom_3x = FSRCNN(factor=3).to(device)
    torch.manual_seed(1024)
    fsrcnn_config = {
        'epochs': 30,
        'save_period': 5,
        'batch_size': 16,
        'checkpoint_dir': SR_path/'result/FSRCNN-100-300',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'DATASET_TYPE': 'diff',
        'low_res': 100,
        'high_res': 300,
        'learning_rate': 1e-3
    }

    models = [fsrcnn_custom_3x]
    configs = [fsrcnn_config]
    # models = [FSRCNN(factor=3).to(device), FSRCNN_Original(
    #     scale_factor=3, num_channels=3).to(device)]
    # models = [ResNetPretrainedDSRes().to(device)]
    # models = [ResNetPretrainedDS().to(device)]
    # models = [FSRCNN(factor=3).to(device)]
    # models = [DRRN().to(device)]
    # models = [VDSR().to(device)]
    DIV2K_path = (SR_path.parent/'datasets'/'DIV2K').absolute()
    DS = DIV2K_Square_Dataset

    train_set_percentage = 0.96

    for i in range(len(models)):
        model = models[i]
        config = configs[i]
        DATASET_TYPE = config['DATASET_TYPE']
        lr_number, hr_number = config['low_res'], config['high_res']
        train_in_dir, train_label_dir = DIV2K_path / DATASET_TYPE / \
            f'train_{lr_number}', DIV2K_path / \
            DATASET_TYPE / f'train_{hr_number}'
        test_in_dir, test_label_dir = DIV2K_path / DATASET_TYPE / \
            f'valid_{lr_number}', DIV2K_path / \
            DATASET_TYPE / f'valid_{hr_number}'
        # log training dataset info
        print("=" * 100)
        print(f"training model: {model.__class__}")
        print("Config:")
        print(config)
        print(f"dataset_type: {DATASET_TYPE}")
        print(f"low resolution: {lr_number}")
        print(f"high resolution: {hr_number}")
        print(f"checkpoint_dir: {config['checkpoint_dir']}")
        print("=" * 100)

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        scheduler = None
        dataset = DS(input_dir=train_in_dir, target_dir=train_label_dir,
                     transform=transforms.ToTensor())
        num_train = int(train_set_percentage * len(dataset))
        num_valid = int(len(dataset) - num_train)
        train_set, valid_set = torch.utils.data.random_split(
            dataset, [num_train, num_valid])
        train_loader = DataLoader(
            dataset=train_set, batch_size=config['batch_size'], shuffle=True)
        valid_loader = DataLoader(
            dataset=valid_set, batch_size=config['batch_size'], shuffle=True)
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
        del train_set, valid_set, dataset, train_loader, valid_loader, optimizer, scheduler

        # run test images
        weight_path = pathlib.Path(config['checkpoint_dir'])/'weights'
        weight_files = sorted(os.listdir(weight_path), key=lambda filename: int(
            re.findall('epoch(\d{1,})\.pth', filename)[0]))
        if len(weight_files) != 0:
            print("Running tests")
            test_all.main(config['DATASET_TYPE'], config['low_res'],
                          config['high_res'], weight_files[-1], config['checkpoint_dir']/'test', model.__class__.__name__)
