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
from model.FSRCNN import FSRCNN
from model.resnet_sr import ResNetPretrainedSS, ResNetPretrainedDS, ResNetPretrainedDSRes

# %%
# setup Dataset
DIV2K_path = pathlib.Path(
    '/home/hacker/Documents/Super-Resolution/datasets/DIV2K').absolute()
train_in_dir, train_label_dir = DIV2K_path / 'diff' / \
    'train_200', DIV2K_path / 'diff' / 'train_600'
test_in_dir, test_label_dir = DIV2K_path / 'diff' / \
    'valid_200', DIV2K_path / 'diff' / 'valid_600'
DS = DIV2K_Square_Dataset

# %% constants and parameters
train_set_percentage = 0.8
config = {
    'epochs': 25,
    'save_period': 5,
    'batch_size': 20,
    'checkpoint_dir': '/home/hacker/Documents/Super-Resolution/SR/result/FSRCNN',
    'log_step': 10,
    'start_epoch': 1
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FSRCNN(scale_factor=3, num_channels=3).to(
    device)  # pass in 600x600 for this model
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam([
    {'params': model.first_part.parameters()},
    {'params': model.mid_part.parameters()},
    {'params': model.last_part.parameters(), 'lr': 1e-3 * 0.1}
], lr=1e-3)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
lr_scheduler = None

# %%
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

# %%
device_name = torch.cuda.get_device_name(
    device) if device.type != 'cpu' else 'cpu'
print(f"Running on {device_name}")
trainer = Trainer(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  device=device,
                  config=config,
                  train_dataset=train_set,
                  valid_dataset=valid_set,
                  train_dataloader=train_loader,
                  valid_dataloader=valid_loader,
                  lr_scheduler=lr_scheduler)
trainer.train()
