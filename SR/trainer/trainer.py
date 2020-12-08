import PIL
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import transforms
from trainer.base_trainer import BaseTrainer
import matplotlib.pyplot as plt
import numpy as np


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, train_dataset, optimizer, device, config: dict,
                 valid_dataset=None,
                 train_dataloader=None,
                 valid_dataloader=None,
                 lr_scheduler=None):
        super(Trainer, self).__init__(model, criterion,
                                      optimizer, train_dataset, valid_dataset, config)
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.do_validation = self.valid_dataloader is not None
        # number of batches in dataloader
        self.len_epoch = len(self.train_dataloader)
        self.log_step = config['log_step']

    def _train_epoch(self, epoch):
        """
        Training logic for one epoch
        :param epoch: Integer, current epoch.
        :return: A log that contains training information of current epoch
        """
        self.model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            inputs, labels = data.to(self.device), target.to(self.device)
            #     del data, target
            output = self.model(inputs)

            loss = self.criterion(output, labels)
            total_loss += loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # update progress bar
            self.progress_bar.update(len(inputs))
            self.progress_bar.set_postfix(train_L=loss.item(),
                                          val_L=(self.valid_loss[-1].item() if len(self.valid_loss) != 0 else None))
        #     del inputs, labels, loss, output
        #     torch.cuda.empty_cache()
        self.train_loss.append(total_loss / len(self.train_dataset))

        # do validation
        valid_loss = self._valid_epoch(epoch) if self.do_validation else None
        self.valid_loss.append(valid_loss)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return {'valid_loss': valid_loss}

    def _valid_epoch(self, epoch):
        """
        Validation after training an epoch
        :param epoch: Integer, current epoch
        :return: A log that contains validation information of current epoch
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_dataloader):
                inputs, labels = data.to(self.device), target.to(self.device)
                del data, target
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                total_loss += loss
                del inputs, labels, output, loss
        return total_loss / len(self.valid_dataset)

    def _progress(self, batch_idx):
        # base = '[{}/{} ({:.0f}%)]'
        # if hasattr(self.train_dataloader, 'n_samples'):
        #     current = batch_idx * self.train_dataloader.batch_size
        #     total = self.train_dataloader.n_samples
        # else:
        #     current = batch_idx
        #     total = self.len_epoch
        # return base.format(current, total, 100.0 * current / total)
        pass

    def _save_checkpoint(self, epoch):
        super(Trainer, self)._save_checkpoint(epoch)
        # save one batch of validation image
        self.model.eval()
        with torch.no_grad():
            data, target = next(iter(self.valid_dataloader))
            data, target = data.to(self.device), target.to(self.device)
            output_path = self.valid_results / f'epoch{epoch}.png'
            # output_path.mkdir(parents=True, exist_ok=False)
            output = self.model(data)
            input_images = [transforms.ToPILImage()(img) for img in data]
            output_images = [transforms.ToPILImage()(img) for img in output]
            target_images = [transforms.ToPILImage()(img) for img in target]
            plot_height = self.valid_dataloader.batch_size * 10
            fig, axes = plt.subplots(
                nrows=len(input_images), ncols=3, figsize=(30, plot_height))
            for i in range(len(input_images)):
                interpolate_img = input_images[i].resize(
                    target.shape[-2:], resample=PIL.Image.BICUBIC)
                axes[i][0].imshow(interpolate_img)
                axes[i][1].imshow(output_images[i])
                axes[i][2].imshow(target_images[i])
            fig.savefig(output_path)
            plt.close()
            del data, target, input_images, output_images, target_images
        np.savetxt(self.checkpoint_dir / 'valid_loss.txt', self.valid_loss)
        np.savetxt(self.checkpoint_dir / 'train_loss.txt', self.train_loss)
