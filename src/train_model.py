import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tensorflow import Tensor
from tqdm import tqdm

from DRRN import DRRN
from VDSR import VDSR

training_target_dir = "./DIV2K_train_HR_crop_600"
training_input_dir = "./DIV2K_train_LR_600_150"
validation_target_dir = "./DIV2K_valid_HR_crop_600"
validation_input_dir = "./DIV2K_valid_LR_600_150"
model_dir = "./VDSR.weight"


def get_training_data(index_start: int = 1, index_end: int = 800):
    training_input = []
    training_output = []
    for i in sorted(os.listdir(training_input_dir))[index_start: index_end + 1]:
        ipt = cv2.imread(training_input_dir + "/" + i, 1)
        opt = cv2.imread(training_target_dir + "/" + i, 1)
        training_input.append(np.stack([ipt[:, :, i] for i in range(2, -1, -1)]))
        training_output.append(np.stack([opt[:, :, i] for i in range(2, -1, -1)]))
    return (torch.from_numpy(np.stack(training_input).astype(np.float32)).cuda(),
            torch.from_numpy(np.stack(training_output).astype(np.float32)).cuda()) \
        if torch.cuda.is_available() else \
        (torch.from_numpy(np.stack(training_input).astype(np.float32)),
         torch.from_numpy(np.stack(training_output).astype(np.float32)))


def get_validation_data(index_start: int = 801, index_end: int = 900):
    validation_input = []
    validation_output = []
    for i in sorted(os.listdir(training_input_dir))[index_start: index_end + 1]:
        ipt = cv2.imread(validation_input_dir + "/" + i, 1)
        opt = cv2.imread(validation_target_dir + "/" + i, 1)
        validation_input.append(np.stack([np.array(ipt)[:, :, i] for i in range(2, -1, -1)]))
        validation_output.append(np.stack([np.array(opt)[:, :, i] for i in range(2, -1, -1)]))
    return (torch.from_numpy(np.stack(validation_input).astype(np.float32)).cuda(),
            torch.from_numpy(np.stack(validation_output).astype(np.float32)).cuda()) \
        if torch.cuda.is_available() else \
        (torch.from_numpy(np.stack(validation_input).astype(np.float32)),
         torch.from_numpy(np.stack(validation_output).astype(np.float32)))


def train_model(mod: nn.Module, batch_size: int = 40, num_epochs: int = 50,
                optimizer=optim.Adam):
    model = mod()
    optimizer = optimizer(model.parameters(), lr=0.005)
    training_loss = []
    validation_loss = [np.nan]

    with tqdm(range(num_epochs), total=num_epochs, file=sys.stdout) as pbar:
        for i in pbar:
            j = 1
            while j < len(os.listdir(training_input_dir)):
                training_input, training_target = get_training_data(j, j + batch_size - 1)
                # in your training loop:
                optimizer.zero_grad()  # zero the gradient buffers
                training_output = model(training_input)
                train_loss = F.mse_loss(training_output, training_target)
                train_loss.backward()
                optimizer.step()
                training_loss.append(train_loss.item())
                del training_input, training_target, training_output, train_loss
                torch.cuda.empty_cache()
                j += batch_size
                pbar.set_postfix(training_loss=training_loss[-1],
                                 validation_loss=validation_loss[-1])
            validation_input, validation_output = get_validation_data()
            valid_opt = model(validation_input)
            valid_loss = F.mse_loss(valid_opt, validation_output)
            validation_loss.append(valid_loss.item())
            del valid_loss, valid_opt, validation_input, validation_output
            torch.cuda.empty_cache()
            pbar.update(1)
    torch.save(model.state_dict(), model_dir + "/{}.pt".format(mod.__name__))
    return model.cpu(), np.array(training_loss), np.array(validation_loss[1:])


def plot_performance(training_loss, validation_loss, model_name: str, batch_size=40):
    plt.figure(figsize=(8, 6))
    plt.plot(800 / batch_size * np.arange(validation_loss.shape[0]), validation_loss, label="Validation")
    plt.plot(np.arange(training_loss.shape[0]), training_loss, label="Training")
    plt.savefig("{}-losses.png".format(model_name))


def plot_predictions(model: nn.Module, test_input: Tensor, test_output: Tensor):
    plt.figure(figsize=(30, 100))
    with torch.no_grad():
        interpolated = F.interpolate(test_input[10:21], scale_factor=4).detach().cpu().numpy()
        valid_opt = model(test_input[10:21]).detach().cpu().numpy()
    for i in range(10):
        plt.subplot(10, 3, 3 * i + 1)
        plt.imshow(np.stack(interpolated[i, :, :, :].astype(np.uint8), axis=2))
        plt.subplot(10, 3, 3 * i + 2)
        plt.imshow(np.stack(valid_opt[i, :, :, :].astype(np.uint8), axis=2))
        plt.subplot(10, 3, 3 * i + 3)
        plt.imshow(np.stack(test_output.detach().cpu().numpy()[i + 10, :, :, :].astype(np.uint8), axis=2))
    plt.savefig("{}-prediction.png".format(model.__class__.__name__))


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        tqdm.write(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        tqdm.write('Running on CPU.')

    validation_input, validation_output = get_validation_data()
    models = [DRRN, VDSR]
    for i in models:
        model, train, valid = train_model(i)
        plot_performance(train, valid, i.__name__)
        plot_predictions(model, validation_input.cpu(), validation_output)
