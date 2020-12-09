# %%
import os
import torch
import pathlib
from PIL import Image
from model.FSRCNN import FSRCNN
from torchvision import transforms

# %%


# %%
model_module = FSRCNN
model = model_module(scale_factor=3, num_channels=3)
model.load_state_dict(torch.load(
    '/home/hacker/Documents/Super-Resolution/SR/result/FSRCNN/weights/epoch_25.pth'))
model.eval()
src_image_path = pathlib.Path(
    '/home/hacker/Documents/Super-Resolution/datasets/DIV2K/diff/train_150/0003.png')
target_image_path = pathlib.Path(
    '/home/hacker/Documents/Super-Resolution/datasets/DIV2K/diff/train_600/0003.png')
output_path = pathlib.Path(
    '/home/hacker/Documents/Super-Resolution/SR/result/FSRCNN/validation')
image = Image.open(src_image_path)

# %%
image_name = os.path.basename(src_image_path)
filename_base, filename_ext = os.path.splitext(image_name)
output_image_name = filename_base + '-output-' + \
    model_module.__name__ + filename_ext
# %%
image_tensor = transforms.ToTensor()(image).unsqueeze(0)
output = model(image_tensor).squeeze(0)
output_image = transforms.ToPILImage()(output)
output_image.save(output_path/output_image_name)
image.save(output_path/image_name)
# %%


# %%
image.close()

# %%
