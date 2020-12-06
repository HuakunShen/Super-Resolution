
import os
from numpy.core.fromnumeric import argsort
import torch
import argparse
import pathlib
import shutil
import numpy as np
import PIL
from fastai.vision import *
from torchvision import transforms


def main(high_res=256, low_res_factor=4, num_extra=2):
    DIV2K_path = pathlib.Path('.').parent.absolute()
    train_HR = DIV2K_path/'DIV2K_train_HR'
    valid_HR = DIV2K_path/'DIV2K_valid_HR'
    assert DIV2K_path.exists() and train_HR.exists() and valid_HR.exists()

    low_res = high_res // low_res_factor
    output_path = DIV2K_path/'custom'/f'{low_res}x{low_res_factor}'
    print(f'high resolution: {high_res}')
    print(f'low resolution: {low_res}')
    print(f'output path: {output_path}')

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    hr_train = output_path/'hr_train'
    hr_valid = output_path/'hr_valid'
    lr_train = output_path/'lr_train'
    lr_valid = output_path/'lr_valid'
    for dir_path in [hr_train, hr_valid, lr_train, lr_valid]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

    hr_valid_image_list = ImageList.from_folder(valid_HR)
    hr_train_image_list = ImageList.from_folder(train_HR)
    hr_train_image_name_list = [img_path.relative_to(
        train_HR) for img_path in hr_train_image_list.items]
    shapes = [PIL.Image.open(
        img_path).size for img_path in hr_train_image_list.items]
    print(
        f"min dimension of all training images={torch.min(torch.tensor(shapes))}")
    hr_valid_image_name_list = [img_path.relative_to(
        valid_HR) for img_path in hr_valid_image_list.items]
    shapes = [PIL.Image.open(
        img_path).size for img_path in hr_valid_image_list.items]
    print(
        f"min dimension of all validation images={torch.min(torch.tensor(shapes))}")

    def get_proper_size_center_crop(image, size: int):
        return transforms.Compose([
            transforms.CenterCrop(min(image.size)),
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC)
        ])(image)

    class ImageAugmentor(object):
        def __init__(self, dest_dir, size: int, num_extra: int = 2):
            assert num_extra <= 5
            self.num_extra = num_extra
            self.dest_dir = dest_dir
            self.size = size
            self.base_image, self.image_name, self.filename_no_ext, self.ext = None, None, None, None
            self.generated_images, self.generated_image_names = [], []

        def generate(self):
            random_choices = np.random.choice(
                a=[1, 2, 3, 4, 5], replace=False, size=self.num_extra)
            for choice in random_choices:
                if choice == 1:
                    self.generated_images.append(
                        transforms.functional.hflip(self.base_image))
                    self.generated_image_names.append(
                        self.filename_no_ext + '-hf' + self.ext)
                elif choice == 2:
                    self.generated_images.append(
                        transforms.functional.vflip(self.base_image))
                    self.generated_image_names.append(
                        self.filename_no_ext + '-vf' + self.ext)
                elif choice == 3:
                    self.generated_images.append(
                        transforms.functional.rotate(self.base_image, 90))
                    self.generated_image_names.append(
                        self.filename_no_ext + '-r90' + self.ext)
                elif choice == 4:
                    self.generated_images.append(
                        transforms.functional.rotate(self.base_image, 180))
                    self.generated_image_names.append(
                        self.filename_no_ext + '-r180' + self.ext)
                elif choice == 5:
                    self.generated_images.append(
                        transforms.functional.rotate(self.base_image, 270))
                    self.generated_image_names.append(
                        self.filename_no_ext + '-r270' + self.ext)

        def save(self):
            for i in range(len(self.generated_image_names)):
                output_path = self.dest_dir/self.generated_image_names[i]
                self.generated_images[i].save(output_path)

        def __call__(self, image_path, i):
            self.image_name = os.path.basename(image_path)
            self.filename_no_ext, self.ext = os.path.splitext(self.image_name)
            self.base_image = get_proper_size_center_crop(
                PIL.Image.open(image_path), self.size)
            self.generated_images = [self.base_image]
            self.generated_image_names = [self.image_name]
            self.generate()
            self.save()

    parallel(ImageAugmentor(hr_valid, high_res, num_extra=0),
             hr_valid_image_list.items)
    parallel(ImageAugmentor(hr_train, high_res, num_extra=num_extra),
             hr_train_image_list.items)

    new_hr_valid_image_list = ImageList.from_folder(hr_valid)
    new_hr_train_image_list = ImageList.from_folder(hr_train)
    new_hr_valid_image_names_list = [img_path.relative_to(
        hr_valid) for img_path in new_hr_valid_image_list.items]
    shapes = [PIL.Image.open(
        img_path).size for img_path in new_hr_valid_image_list.items]
    print(
        f"min dimension of all generated HR training images={torch.min(torch.tensor(shapes))}")
    new_hr_train_image_names_list = [img_path.relative_to(
        hr_train) for img_path in new_hr_train_image_list.items]
    shapes = [PIL.Image.open(
        img_path).size for img_path in new_hr_train_image_list.items]
    print(
        f"min dimension of all generated HR validation images={torch.min(torch.tensor(shapes))}")

    class Downscaler(object):
        def __init__(self, src_path, dest_path, low_res_size):
            self.src_path = src_path
            self.dest_path = dest_path
            self.low_res_size = low_res_size

        def __call__(self, image_name, i):
            src_image_path = self.src_path/image_name
            target_image_path = self.dest_path/image_name
            src_img = PIL.Image.open(src_image_path)
            downscaled_img = src_img.resize(
                (self.low_res_size, self.low_res_size), resample=PIL.Image.BILINEAR).convert('RGB')
            downscaled_img.save(target_image_path)

    parallel(Downscaler(hr_valid, lr_valid, low_res),
             new_hr_valid_image_names_list)
    parallel(Downscaler(hr_train, lr_train, low_res),
             new_hr_train_image_names_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Images for Super Resolution')
    parser.add_argument('--high_res', default=600, type=int,
                        help='high resolution image size')
    parser.add_argument('-f', '--factor', default=4, type=int,
                        help='factor to scale down high res image to low res image')
    parser.add_argument('-n', '--num_extra', default=2, type=int,
                        help='number of extra variant to add')
    args = parser.parse_args()
    main(high_res=args.high_res, low_res_factor=args.factor,
         num_extra=args.num_extra)
