import os
import PIL
import torch
import shutil
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from torchvision import transforms
from model.FSRCNN import FSRCNN, FSRCNN_Original


"""
Example:
python3 test_all.py --model FSRCNN -t diff --low 200 --high 600 -w /home/hacker/Documents/Super-Resolution/SR/result/FSRCNN_new/weights/epoch25.pth -o /home/hacker/Documents/Super-Resolution/SR/result/FSRCNN_new/test
"""

model_class_map = {
    'FSRCNN': FSRCNN(factor=3),
    'FSRCNN_Original': FSRCNN_Original(scale_factor=3, num_channels=3)
}


class Saver():
    def __init__(self, output_path, valid_lr, valid_hr, model) -> None:
        self.output_path = output_path
        self.valid_lr = valid_lr
        self.valid_hr = valid_hr
        self.model = model

    def __call__(self, image_name):
        lr_image_path = self.valid_lr/image_name
        hr_image_path = self.valid_hr/image_name
        assert lr_image_path.exists() and hr_image_path.exists()
        lr_image, hr_image = PIL.Image.open(
            lr_image_path), PIL.Image.open(hr_image_path)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
        computed_image = transforms.ToPILImage()(
            self.model(transforms.ToTensor()(lr_image).unsqueeze(0)).squeeze(0))
        axes[0].imshow(lr_image.resize(hr_image.size),
                       resample=PIL.Image.BICUBIC)
        axes[1].imshow(computed_image)
        axes[2].imshow(hr_image)
        for i in range(3):
            axes[i].axis('off')
        lr_image.close()
        hr_image.close()
        fig.savefig(self.output_path/f"test-{image_name}.png")


def main(type_, lr_size, hr_size, weight_path, output_path, model_name):
    weight_path = pathlib.Path(weight_path)
    output_path = pathlib.Path(output_path)
    model = model_class_map[model_name]
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    DIV2K_DIR = pathlib.Path('../datasets/DIV2K')
    image_dir = DIV2K_DIR/type_
    valid_lr = image_dir/f'valid_{lr_size}'
    valid_hr = image_dir/f'valid_{hr_size}'
    assert DIV2K_DIR.exists() and image_dir.exists(
    ) and valid_lr.exists() and valid_hr.exists()

    model.load_state_dict(torch.load(weight_path))
    model.eval()
    image_names = os.listdir(valid_lr)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        with tqdm(total=len(image_names)) as pbar:
            for i, y in enumerate(p.imap_unordered(Saver(output_path, valid_lr, valid_hr,
                                                         model), image_names)):
                pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('test all arg parser')
    parser.add_argument("-t", "--type", help="can be either same or diff")
    parser.add_argument("--low", help="low resolution")
    parser.add_argument("--high", help="high resolution")
    parser.add_argument("-w", "--weight", help="weight path")
    parser.add_argument("-o", "--output", help="output path")
    parser.add_argument("-m", "--model", help="model name")
    args = parser.parse_args()
    main(args.type, args.low, args.high, args.weight, args.output, args.model)