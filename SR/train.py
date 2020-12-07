import torch
import pathlib
import torchvision
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from trainer.trainer import Trainer
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataloader.datasets import DIV2K_Square_Dataset
from model.FSRCNN import FSRCNN, FSRCNN_Original
from model.resnet_sr import ResNetPretrainedSS, ResNetPretrainedDS, ResNetPretrainedDSRes


def train(mod: nn.Module, criterion, optimizer_, device_, config_, train_dataset, valid_dataset, train_dataloader,
          valid_dataloader, scheduler_lr):
    device_name = torch.cuda.get_device_name(
        device_) if device_.type != 'cpu' else 'cpu'
    print(f"Running on {device_name}")
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
    # setup Dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # models = [FSRCNN(factor=3).to(device), FSRCNN_Original(
    #     scale_factor=3, num_channels=3).to(device)]
    models = [ResNetPretrainedDSRes().to(device)]
    configs = [
        {
            'epochs': 25,
            'save_period': 5,
            'batch_size': 5,
            'checkpoint_dir': '/home/hacker/Documents/Super-Resolution/SR/result/ResNetPretrainedDSRes',
            'log_step': 10,
            'start_epoch': 1,
            'criterion': nn.MSELoss()
        },

    ]

    DIV2K_path = pathlib.Path('../datasets/DIV2K').absolute()
    train_in_dir, train_label_dir = DIV2K_path / 'diff' / \
        'train_150', DIV2K_path / 'diff' / 'train_600'
    test_in_dir, test_label_dir = DIV2K_path / 'diff' / \
        'valid_150', DIV2K_path / 'diff' / 'valid_600'
    DS = DIV2K_Square_Dataset

    train_set_percentage = 0.8

    for i in range(len(models)):
        model = models[i]
        config = configs[i]

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # scheduler = None
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
