# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import PIL
import torch
import shutil
import pathlib
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from model.FSRCNN import FSRCNN
from torchvision import transforms


# %%
lr_size = 200
hr_size = 600
type_ = 'diff'
weight_path = '/home/hacker/Documents/Super-Resolution/SR/result/FSRCNN/weights/epoch25.pth'
output_path = pathlib.Path(
    '/home/hacker/Documents/Super-Resolution/SR/result/FSRCNN/test')
model_class = FSRCNN


# %%
if output_path.exists():
    shutil.rmtree(output_path)
output_path.mkdir(parents=True, exist_ok=False)
DIV2K_DIR = pathlib.Path('../datasets/DIV2K')
image_dir = DIV2K_DIR/type_
valid_lr = image_dir/f'valid_{lr_size}'
valid_hr = image_dir/f'valid_{hr_size}'
assert DIV2K_DIR.exists() and image_dir.exists(
) and valid_lr.exists() and valid_hr.exists()


# %%
model = model_class(scale_factor=3, num_channels=3)
model.load_state_dict(torch.load(weight_path))
model.eval()


# %%
image_names = os.listdir(valid_lr)


# %%
def save_image(image_name):
    lr_image_path = valid_lr/image_name
    hr_image_path = valid_hr/image_name
    assert lr_image_path.exists() and hr_image_path.exists()
    lr_image, hr_image = PIL.Image.open(
        lr_image_path), PIL.Image.open(hr_image_path)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
    computed_image = transforms.ToPILImage()(
        model(transforms.ToTensor()(lr_image).unsqueeze(0)).squeeze(0))
    axes[0].imshow(lr_image.resize(hr_image.size), resample=PIL.Image.BICUBIC)
    axes[1].imshow(computed_image)
    axes[2].imshow(hr_image)
    for i in range(3):
        axes[i].axis('off')
    lr_image.close()
    hr_image.close()
    fig.savefig(output_path/f"test-{image_name}.png")


# %%
with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    p.map(save_image, image_names)


# %%
