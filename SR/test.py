import os
import pathlib
import argparse
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.util import psnr, get_divider_str, psnr_PIL
import logging
import matplotlib.pyplot as plt
import torch
from dataloader.datasets import SRDataset
from torch.utils.data import DataLoader

"""
Single Image Example:
python test.py -w '/home/hacker/Insync/shenhuakun@outlook.com/OneDrive/UT/4th Year/CSC420/project/Super-Resolution-Results/TEXT/unetd4-blur-7-MSE/model.pth' -i /home/hacker/Documents/Super-Resolution/datasets/TEXT/same/valid_BlurRadius7/1948.bmp -o /home/hacker/Desktop/output.png

No Target Batch Test
python test.py -w '/home/hacker/Insync/shenhuakun@outlook.com/OneDrive/UT/4th Year/CSC420/project/Super-Resolution-Results/TEXT/unetd4-blur-7-MSE/model.pth' -i /home/hacker/Documents/Super-Resolution/datasets/TEXT/same/valid_BlurRadius7 -o /home/hacker/Desktop/output

With Target Batch Test
python test.py -w '/home/hacker/Insync/shenhuakun@outlook.com/OneDrive/UT/4th Year/CSC420/project/Super-Resolution-Results/TEXT/unetd4-blur-7-MSE/model.pth' -i /home/hacker/Documents/Super-Resolution/datasets/TEXT/same/valid_BlurRadius7 -o /home/hacker/Desktop/output -t /home/hacker/Documents/Super-Resolution/datasets/TEXT/same/valid_Target300x300
"""

# setup logger
logging.basicConfig(level=logging.INFO, filename='test.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def main_no_target(in_path: pathlib.Path, output_dir: pathlib.Path, weight_path: pathlib.Path, batch_size=10):
    model = torch.load(weight_path)
    image_names = sorted(os.listdir(in_path))
    for image_name in tqdm(image_names):
        img = Image.open(in_path / image_name)
        computed_img = compute(model, img)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        axes[0].imshow(np.asarray(img.resize(computed_img.size)))
        axes[0].set_title("Input Image", fontsize=20)
        axes[1].imshow(np.asarray(computed_img))
        axes[1].set_title("Computed Image", fontsize=20)
        output_path = output_dir / (image_name.split('.')[0] + ".png")
        fig.savefig(output_path)
        img.close()
        plt.close()


def main_with_target(in_dir, out_dir, label_path, weight_path, batch_size=10):
    model = torch.load(weight_path)
    model = model.to(device)
    model.eval()
    image_names = sorted(os.listdir(in_dir))
    low_res_psnr_sum = 0
    computed_psnr_sum = 0
    dataset = SRDataset(input_dir=in_dir, target_dir=label_path, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True)
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        for i, out in enumerate(output):
            out_name = f"{batch_idx * batch_size + i}.png"
            input_image = transforms.ToPILImage()(data[i]).resize(target.shape[2:])
            computed_img = transforms.ToPILImage()(out)
            label_img = transforms.ToPILImage()(target[i])
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
            low_res_psnr = psnr_PIL(input_image, label_img)
            low_res_psnr_sum += low_res_psnr
            computed_psnr = psnr_PIL(computed_img, label_img)
            computed_psnr_sum += computed_psnr
            axes[0].set_title(
                f"Low Res (PSNR: {round(float(low_res_psnr), 2)})", fontsize=20)
            axes[0].imshow(np.asarray(input_image.resize(computed_img.size)))
            axes[1].imshow(np.asarray(computed_img))  # generated image
            axes[1].set_title(
                f"Generated (PSNR: {round(float(computed_psnr), 2)})", fontsize=20)
            axes[2].imshow(np.asarray(label_img))  # high resolution image
            axes[2].set_title(f"High Res Image", fontsize=20)
            for i in range(3):
                axes[i].axis('off')
            output_path = out_dir / out_name
            fig.savefig(output_path)
            plt.close()

            label_img.close()

    low_res_psnr_avg = low_res_psnr_sum / len(image_names)
    computed_psnr_avg = computed_psnr_sum / len(image_names)

    psnr_txt = str(out_dir.parent / 'psnr.txt')
    with open(psnr_txt, 'w') as f:
        f.writelines([
            f"low_res_psnr_avg: {low_res_psnr_avg}\n",
            f"computed_psnr_avg: {computed_psnr_avg}\n"
        ])


def single_image(in_path, out_path, weight_path):
    in_img = Image.open(in_path)
    model = torch.load(weight_path)
    output_image = compute(model, in_img)
    output_image.save(out_path)
    in_img.close()


def compute(model, in_img: Image):
    return transforms.ToPILImage()(model(transforms.ToTensor()(in_img).unsqueeze(0)).squeeze(0))


if __name__ == "__main__":
    logger.info(get_divider_str("Begin"))

    parser = argparse.ArgumentParser()
    # parser.add_argument("-b", "--batch", action='store_true', help("Whether you want a batch test (multiple image)"))
    parser.add_argument(
        "-i", "--input", help="Either a single image file or a directory of images", required=True)

    parser.add_argument("-o", "--output",
                        help="output directory (multiple input image) or filename if input is a single image",
                        required=True)
    parser.add_argument(
        "-l", "--label", help="target directory containing target images corresponding to input directory")
    parser.add_argument(
        "-w", "--weight", help="Weight File Path", required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=10, help="batch size for GPU Running test")
    args = parser.parse_args()
    in_path = pathlib.Path(args.input).absolute()
    out_path = pathlib.Path(args.output).absolute()
    label_path = pathlib.Path(args.label).absolute(
    ) if args.label is not None else None
    weight_path = pathlib.Path(args.weight).absolute()
    if not (in_path.exists() and weight_path.exists()) or (
            args.label is not None and not label_path.exists()):
        raise ValueError("invalid input, one or more paths doesn't exist")
    if not out_path.exists():
        if not os.path.isfile(in_path):
            out_path.mkdir(parents=True, exist_ok=True)
    else:
        logger.warning(
            f"output path {out_path} exists, your output directory may be removed")

    if os.path.isfile(in_path):
        single_image(in_path, out_path, weight_path)
    elif os.path.isdir(in_path):
        if label_path is None:
            main_no_target(in_path, out_path, weight_path)
        else:
            main_with_target(in_path, out_path, label_path, weight_path, args.batch_size)
        logger.info(get_divider_str("Done"))
