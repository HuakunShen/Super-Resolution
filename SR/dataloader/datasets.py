import os
import re
import sys
import torchvision
import torch
import pathlib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
from tqdm import tqdm

class DIV2K_Square_Dataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str, transform=ToTensor()) -> None:
        # verify data integrity
        input_files, target_files = os.listdir(
            input_dir), os.listdir(target_dir)
        self.lr_filenames = sorted(input_files)
        self.hr_filenames = sorted(target_files)
        self.filenames = self.hr_filenames
        self.transform = transform
        self.input_dir = os.path.abspath(input_dir)
        self.target_dir = os.path.abspath(target_dir)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index):
        """
        :param index:
        :return: a input image (low resolution) and a target image (high resolution)
        """
        input_img = Image.open(os.path.join(
            self.input_dir, self.lr_filenames[index]))
        target_img = Image.open(os.path.join(
            self.target_dir, self.hr_filenames[index]))
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img


if __name__ == '__main__':
    # sample code
    DIV2K_path = pathlib.Path('/home/hacker/Documents/Super-Resolution/datasets/DIV2K').absolute()
    hr_train = DIV2K_path / 'same' / 'train_600'
    hr_valid = DIV2K_path / 'same' / 'valid_600'
    lr_train = DIV2K_path / 'same' / 'train_150'
    lr_valid = DIV2K_path / 'same' / 'valid_150'
    datasets = {
        'train': DIV2K_Square_Dataset(input_dir=lr_train, target_dir=hr_train, transform=transforms.ToTensor()),
        'val': DIV2K_Square_Dataset(input_dir=lr_valid, target_dir=hr_valid, transform=transforms.ToTensor())
    }
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
                   ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    print(dataloaders['train'].batch_size)

    # for batch_idx, (data, target) in enumerate(tqdm(dataloaders['val'])):
    #     pass
    #
    with tqdm(dataloaders['val'], file=sys.stdout) as pbar:
        for i, (data, label) in enumerate(pbar):
            pbar.set_postfix(training_loss=i, validation_loss=i)






