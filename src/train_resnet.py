# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from datetime import datetime

import PIL
import time
import copy
import torch
import pathlib
import torchvision
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model.model import ResNetSR
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from custom_dataset import DIV2K_Dataset, DIV2K_Square_Dataset

# %%
DIV2K_path = pathlib.Path('../datasets/DIV2K').absolute()
DIV2K_path

# %%
DIV2K_train_HR_crop_600 = DIV2K_path / 'custom' / 'DIV2K_train_HR_crop_600'
DIV2K_train_LR_600_150 = DIV2K_path / 'custom' / 'DIV2K_train_LR_600_150'
DIV2K_valid_HR_crop_600 = DIV2K_path / 'custom' / 'DIV2K_valid_HR_crop_600'
DIV2K_valid_LR_600_150 = DIV2K_path / 'custom' / 'DIV2K_valid_LR_600_150'

# %%
train_LR, train_HR = DIV2K_train_LR_600_150, DIV2K_train_HR_crop_600
valid_LR, valid_HR = DIV2K_valid_LR_600_150, DIV2K_valid_HR_crop_600

# %%
datasets = {
    'train': DIV2K_Square_Dataset(input_dir=train_LR, target_dir=train_HR, transform=transforms.ToTensor()),
    'val': DIV2K_Square_Dataset(input_dir=valid_LR, target_dir=valid_HR, transform=transforms.ToTensor())
}
dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# Low Res
inputs, target = next(iter(dataloaders['train']))
transforms.ToPILImage()(torchvision.utils.make_grid(inputs))

# %%
# High Res
transforms.ToPILImage()(torchvision.utils.make_grid(target))

# %%
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            t0 = time.time()

            # Iterate over data.
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                elapsed = time.time() - t0
                if step % 40 == 0 and not step == 0:
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(dataloaders[phase]), elapsed))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                # PyTorch accumulates the gradients on subsequent backward passes (useful for RNN)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    print(f"output shape: {outputs.shape}")
                    # _, preds = torch.max(outputs, 1)
                    # print(f"preds shape: {preds.shape}")
                    print(outputs.shape, labels.shape)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # print(f"preds shape: {preds.shape}")

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print(preds.shape)
                # print(labels.shape)
                # print(labels.data.shape)
                # running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train(model, criterion, optimizer, scheduler, num_epochs=25):
    loss_values = []
    for epoch_i in range(0, num_epochs):
        print("")
        print(
            '======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        model.train()
        for step, (inputs, labels) in enumerate(dataloaders['train']):
            if step % 40 == 0 and not step == 0:
                elapsed = time.time() - t0
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(dataloaders['train']), elapsed))
            model.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) * inputs.shape[0]
            total_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / dataset_sizes['train']
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(time.time() - t0))

    print("Running Validation...")
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        # Evaluate data for one epoch
        for step, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) * inputs.shape[0]

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")
    return loss_values


model = ResNetSR(factor=4)
model = model.to(device)
criterion = nn.MSELoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# %%
loss_values = train(model, criterion, optimizer_ft,
                    exp_lr_scheduler, num_epochs=2)
print(loss_values)
