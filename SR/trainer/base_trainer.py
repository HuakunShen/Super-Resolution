from multiprocessing import Value
import os
import sys
import numpy as np
import torch
import pathlib
import shutil
import logging
from numpy import inf
from tqdm import tqdm
from abc import abstractmethod
import matplotlib.pyplot as plt
import test_all
from utils import util
from logger.memory_profile import MemoryProfiler


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
        self.logger = logging.getLogger(os.path.basename(
            pathlib.Path(config["checkpoint_dir"]).absolute()))
        if self.start_epoch == 0:
            raise ValueError("start_epoch must start from at least 1")
        self.progress_bar = None
        self.checkpoint_dir = pathlib.Path(config['checkpoint_dir'])
        self.model_weights_dir = self.checkpoint_dir / 'weights'
        self.valid_results = self.checkpoint_dir / 'validation'
        self.log_path = self.checkpoint_dir / 'log.log'
        self.memory_profiler = MemoryProfiler(self.logger)
        # load valid_loss and train_loss if this is not starting from the beginning
        if self.start_epoch != 1:
            self.logger.info("loadding loss files with numpy")
            self.train_loss = list(np.loadtxt(
                self.checkpoint_dir / 'valid_loss.txt'))
            self.valid_loss = list(np.loadtxt(
                self.checkpoint_dir / 'train_loss.txt'))
            self.logger.debug(
                f'loaded training loss from previous train (length={len(self.train_loss)}):')
            self.logger.debug(self.train_loss)
            self.logger.debug(
                f'loaded validation loss from previous train (length={len(self.valid_loss)}):')
            self.logger.debug(self.valid_loss)
        else:
            self.train_loss = []
            self.valid_loss = []

        # # load checkpoint weights if needed
        # if self.start_epoch <= 1 and self.checkpoint_dir.exists():
        #     # not training from a checkpoint
        #     shutil.rmtree(self.checkpoint_dir)
        if not (self.start_epoch <= 1 and self.checkpoint_dir.exists()):
            # load weights
            self.logger.info(
                f"'start_epoch' is not 1, looking for epoch{self.start_epoch}.pth")
            weights_files = os.listdir(self.model_weights_dir)
            if f'epoch{self.start_epoch}.pth' in weights_files:
                self.logger.info(
                    f"epoch{self.start_epoch}.pth found, load model weights")
                self.model.load_state_dict(torch.load(
                    self.model_weights_dir / f'epoch{self.start_epoch}.pth'))
                self.model.eval()
            else:
                raise ValueError(
                    f"Weight file not found, the start epoch is {self.start_epoch}, epoch{self.start_epoch}.pth doesn't exist in {self.model_weights_dir}")
        for path in [self.checkpoint_dir, self.model_weights_dir, self.valid_results]:
            path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        with tqdm(total=len(self.train_dataset) * (self.epochs - self.start_epoch + 1)) as progress_bar:
            # with tqdm(range(self.start_epoch, self.epochs + 1), total=self.epochs, file=sys.stdout) as progress_bar:
            self.progress_bar = progress_bar
            for epoch in range(self.start_epoch, self.epochs + 1):
                self.progress_bar.set_description(
                    'epoch: {}/{}'.format(epoch, self.epochs))
                result = self._train_epoch(epoch)
                if epoch % self.save_period == 0 or epoch == self.epochs:
                    self._save_checkpoint(epoch)
                self._update_loss_plot()
                self.memory_profiler.update_n_log(epoch)
        self.progress_bar.close()
        self.memory_profiler.log_final_message()

    def _update_loss_plot(self):
        # training loss plot
        if len(self.train_loss) != 0:
            self.logger.debug("plot training loss")
            plt.figure()
            # plt.plot(list(range(self.start_epoch, self.start_epoch +
            #                     len(self.train_loss))), self.train_loss)
            plt.plot(list(range(1, len(self.train_loss) + 1)), self.train_loss)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Training Loss")
            plt.savefig(self.checkpoint_dir / 'train_loss.png')
            plt.close()
        else:
            self.logger.error("error: no training loss")
        if len(self.valid_loss) != 0:
            self.logger.debug("plot validation loss")
            plt.figure()
            # plt.plot(list(range(self.start_epoch, self.start_epoch +
            #                     len(self.valid_loss))), self.valid_loss)
            plt.plot(list(range(1, len(self.valid_loss) + 1)), self.valid_loss)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Validation Loss")
            plt.savefig(self.checkpoint_dir / 'valid_loss.png')
            plt.close()
        else:
            self.logger.error("error: no validation loss")

    def _save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(
            self.model_weights_dir, 'epoch{}.pth'.format(epoch)))
