import os
import re
import torch
import tarfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms


class BSDS_Dataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str, transform=ToTensor()) -> None:
        assert set(os.listdir(input_dir)) == set(os.listdir(target_dir))
        self.filenames = sorted(os.listdir(input_dir))
        self.transform = transform
        self.input_dir = os.path.abspath(input_dir)
        self.target_dir = os.path.abspath(target_dir)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index):
        input_img = Image.open(os.path.join(self.input_dir, self.filenames[index]))
        target_img = Image.open(os.path.join(self.target_dir, self.filenames[index]))
        if input_img.size[0] < input_img.size[1]:
            input_img = input_img.transpose(Image.ROTATE_90)
            target_img = target_img.transpose(Image.ROTATE_90)
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img


class DIV2K_Dataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str, transform=ToTensor()) -> None:
        # verify data integrity
        input_files, target_files = os.listdir(input_dir), os.listdir(target_dir)
        self.lr_filenames = sorted(input_files)
        self.hr_filenames = sorted(target_files)
        self.filenames = self.hr_filenames
        input_files_filtered = ['{}.png'.format(re.findall('(\d{4}).{2}\.png', filename)[0])
                                for filename in self.lr_filenames]
        assert input_files_filtered == self.hr_filenames
        self.low_res_suffix = re.findall('\d{4}(.{2})\.png', input_files[0])[0]
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
        input_img = Image.open(os.path.join(self.input_dir, self.lr_filenames[index]))
        target_img = Image.open(os.path.join(self.target_dir, self.hr_filenames[index]))
        # if input_img.size[0] < input_img.size[1]:
        #     input_img = input_img.transpose(Image.ROTATE_90)
        #     target_img = target_img.transpose(Image.ROTATE_90)
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img


if __name__ == "__main__":
    # BSDS Dataset Demo
    # verify dimension
    dataset = BSDS_Dataset('../BSDS100', '../BSDS100')
    f_shape = dataset[0][0].shape
    for i in range(len(dataset)):
        assert dataset[i][0].shape == f_shape
        assert dataset[i][1].shape == f_shape
    # sample usage
    batch_size = 5
    train_set_percentage = 0.7
    dataset = BSDS_Dataset(input_dir='../BSDS100', target_dir='../BSDS100', transform=transforms.ToTensor())
    num_train = int(train_set_percentage * len(dataset))
    num_test = int(len(dataset) - num_train)
    train_set, test_set = torch.utils.data.random_split(
        dataset, [num_train, num_test])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # DIV2K Dataset Demo
    dataset = DIV2K_Dataset('../datasets/DIV2K/DIV2K_train_LR_x8', '../datasets/DIV2K/DIV2K_train_HR')
