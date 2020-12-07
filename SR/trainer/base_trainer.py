import os
import sys
import torch
import pathlib
import shutil
from numpy import inf
from tqdm import tqdm
from abc import abstractmethod
import matplotlib.pyplot as plt


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
                # valid_loss = result['valid_loss']
                # evaluate model performance according to configured metric, save best checkpoint as model_best
                # TODO: implement the function to early stop if model not improved over certain epochs
                # best = False
                # if self.mnt_mode != 'off':
                #     try:
                #         # check whether model performance improved or not, according to specified metric(mnt_metric)
                #         improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                #                    (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                #     except KeyError:
                #         self.logger.warning("Warning: Metric '{}' is not found. "
                #                             "Model performance monitoring is disabled.".format(self.mnt_metric))
                #         self.mnt_mode = 'off'
                #         improved = False
                #
                #     if improved:
                #         self.mnt_best = log[self.mnt_metric]
                #         not_improved_count = 0
                #         best = True
                #     else:
                #         not_improved_count += 1
                #
                #     if not_improved_count > self.early_stop:
                #         self.logger.info("Validation performance didn\'t improve for {} epochs. "
                #                          "Training stops.".format(self.early_stop))
                #         break

                if epoch % self.save_period == 0 or epoch == self.epochs:
                    self._save_checkpoint(epoch)
                    pass

        # training loss plot
        if len(self.train_loss) != 0:
            plt.figure()
            plt.plot(list(range(1, len(self.train_loss) + 1)), self.train_loss)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Training Loss")
            plt.savefig(self.checkpoint_dir / 'train_loss.png')
        else:
            print("error: no training loss")
        if len(self.valid_loss) != 0:
            plt.figure()
            plt.plot(list(range(1, len(self.valid_loss) + 1)), self.valid_loss)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Validation Loss")
            plt.savefig(self.checkpoint_dir / 'valid_loss.png')
        else:
            print("error: no validation loss")

    def _save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(
            self.model_weights_dir, 'epoch{}.pth'.format(epoch)))
