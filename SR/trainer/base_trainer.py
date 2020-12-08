import os
import sys
import torch
import pathlib
import shutil
from numpy import inf
from tqdm import tqdm
from abc import abstractmethod
import matplotlib.pyplot as plt
import test_all


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, optimizer, train_dataset, valid_dataset, config: dict):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = config['epochs']
        self.save_period = config['save_period']
        self.start_epoch = config['start_epoch']
        self.progress_bar = None
        self.train_loss = []
        self.valid_loss = []
        self.checkpoint_dir = pathlib.Path(config['checkpoint_dir'])
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self.model_weights_dir = self.checkpoint_dir / 'weights'
        self.valid_results = self.checkpoint_dir / 'validation'
        for path in [self.checkpoint_dir, self.model_weights_dir, self.valid_results]:
            path.mkdir(parents=True, exist_ok=False)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        with tqdm(total=len(self.train_dataset) * self.epochs) as progress_bar:
            # with tqdm(range(self.start_epoch, self.epochs + 1), total=self.epochs, file=sys.stdout) as progress_bar:
            self.progress_bar = progress_bar
            for epoch in range(self.start_epoch, self.epochs + 1):
                self.progress_bar.set_description(
                    'epoch: {}/{}'.format(epoch, self.epochs))
                result = self._train_epoch(epoch)
                if epoch % self.save_period == 0 or epoch == self.epochs:
                    self._save_checkpoint(epoch)

        self.progress_bar.close()

    def _save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(
            self.model_weights_dir, 'epoch{}.pth'.format(epoch)))
        # training loss plot
        if len(self.train_loss) != 0:
            plt.figure()
            plt.plot(list(range(1, len(self.train_loss) + 1)), self.train_loss)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Training Loss")
            plt.savefig(self.checkpoint_dir / 'train_loss.png')
            plt.close()
        else:
            print("error: no training loss")
        if len(self.valid_loss) != 0:
            plt.figure()
            plt.plot(list(range(1, len(self.valid_loss) + 1)), self.valid_loss)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Validation Loss")
            plt.savefig(self.checkpoint_dir / 'valid_loss.png')
            plt.close()
        else:
            print("error: no validation loss")
