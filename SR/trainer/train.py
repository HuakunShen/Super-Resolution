import os
import re
from typing import List
import torch
import shutil
import pathlib
import traceback
from torch import nn
import multiprocessing
import torch.optim as optim
from trainer.trainer import Trainer
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataloader.datasets import SRDataset
from model.FSRCNN import FSRCNN, FSRCNN_Original
from model.resnet_sr import ResNetPretrainedSS
from model.SRCNN import SRCNN
from model.UNetSR import UNetSR
from utils.util import get_divider_str
from tests import test_all
from logger import get_logger
from config import SR_PATH, DIV2K_DATASET_PATH


def run(models: List[nn.Module], configs: List[dict]):
    dataset_class = SRDataset
    for i in range(len(models)):
        torch.cuda.empty_cache()
        model = models[i]
        config = configs[i]
        DATASET_PATH = config['dataset']
        ############################### setup workspace ##############################
        if config['start_epoch'] <= 1 and config["checkpoint_dir"].exists() and not config['test_only']:
            # not training from a checkpoint
            shutil.rmtree(config["checkpoint_dir"])
        ################################ setup logger ################################
        config["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
        logger = get_logger(os.path.basename(pathlib.Path(
            config["checkpoint_dir"]).absolute()), config["checkpoint_dir"] / 'log.log')
        ################################## test only #################################
        if config['test_only']:
            logger.info(
                "test_only is True, skip training and produce test images")
            run_test(config, logger, model)
            continue
        ############################# setup dataset path #############################
        lr_type, hr_type = config['low_res'], config['high_res']
        train_in_dir, train_label_dir = DATASET_PATH / config['dataset_type'] / \
            f'train_{lr_type}', DATASET_PATH / \
            config['dataset_type'] / \
            f'train_{hr_type}'
        test_in_dir, test_label_dir = DATASET_PATH / config['dataset_type'] / \
            f'valid_{lr_type}', DATASET_PATH / \
            config['dataset_type'] / \
            f'valid_{hr_type}'
        ########################## log training dataset info ##########################
        logger.info(get_divider_str("Configuration Parameters Start"))
        logger.info(f"training model: {model.__class__}")
        for key in config:
            logger.info(f"{key}: {config[key]}")
        logger.info(get_divider_str("Configuration Parameters End"))

        optimizer = config['optimizer']
        if 'scheduler' in config and config['scheduler'] is not None:
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=config['scheduler']['step_size'], gamma=config['scheduler']['gamma'])
        else:
            scheduler = None
        dataset = dataset_class(input_dir=train_in_dir, target_dir=train_label_dir,
                                transform=transforms.ToTensor())
        num_train = int(config['train_set_percentage'] * len(dataset))
        num_valid = int(len(dataset) - num_train)
        train_set, valid_set = torch.utils.data.random_split(
            dataset, [num_train, num_valid], generator=torch.Generator().manual_seed(420))
        train_loader = DataLoader(
            dataset=train_set, batch_size=config['batch_size'],
            shuffle=True, num_workers=config['num_worker'],
            pin_memory=True)
        valid_loader = DataLoader(
            dataset=valid_set, batch_size=config['batch_size'],
            shuffle=True, num_workers=config['num_worker'],
            pin_memory=True)
        try:
            device_name = torch.cuda.get_device_name(
                config['device']) if config['device'].type != 'cpu' else 'cpu'
            logger.info(f"Running on {device_name}\n")
            trainer = Trainer(model=model,
                              criterion=config['criterion'],
                              optimizer=optimizer,
                              device=config['device'],
                              config=config,
                              train_dataset=train_set,
                              valid_dataset=valid_set,
                              train_dataloader=train_loader,
                              valid_dataloader=valid_loader,
                              lr_scheduler=scheduler)
            trainer.train()

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(e)
            logger.error(error_traceback)
            logger.error("Failed! Skip to next model if there is any left.")
            continue

        ############################################ run test images ############################################
        run_test(config, logger, model)


def run_test(config, logger, model):
    try:
        weight_path = pathlib.Path(config['checkpoint_dir']) / 'weights'
        weight_files = list(filter(lambda filename: re.match(
            'epoch(\d{1,})\.pth', filename), os.listdir(weight_path)))
        weight_files = sorted(weight_files, key=lambda filename: int(
            re.findall('epoch(\d{1,})\.pth', filename)[0]))
        if len(weight_files) != 0:
            logger.info("Running tests")
            test_all.main(
                config['dataset_type'],
                config['low_res'],
                config['high_res'],
                weight_path / weight_files[-1],
                config['dataset'],
                config['checkpoint_dir'] /
                'test', model.__class__.__name__,
                logger=logger,
                multiprocess_num_cpu=1 if not 'test_all_multiprocess_cpu' in config else config[
                    'test_all_multiprocess_cpu']
            )
    except KeyError as e:
        error_traceback = traceback.format_exc()
        logger.error(e)
        logger.error(error_traceback)
        logger.error(
            "Test images failed to generate. Likely your model is not registered in the test_all.py file. Modify the map dictionary called model_map")
