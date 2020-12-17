# SR

This directory **SR** contains a training framework for training and testing.

I will be calling is **SR** for convenience.

Here are the instructions of how to use it.

## Introduction

The entry point is a python training file which you configure and execute for training.

[train_template.py](./train_template.py) is a template for training.
- See [Training](#training) for details of training

[test_template.py](./test_template.py) is a template for testing.
- See [Testing](#testing) for details of testing

By training with SR, the entire training process will be saved
- A **log.log** file containing all training information including will be generated within the checkpoint directory. It contains all parameters for training, debug & error messages, Peak Memory Usage for GPU and RAM. From this file, you can recreate the training configuration dictionary.
- **loss.png** contains the training and validation loss function. From this image you will know if the loss converges and decide whether you need to resume training or stop training.
- **weights** contains pytorch weight files saved periodically (period defined by *'save_period'* parameter in training configuration). With weight files, you can resume training from any point or generate test results.
- **validation** contains partial validation results saved periodically (period defined by *'save_period'* parameter in training configuration). From these validation results, you can view the progress of training more intuitively and decide if you want early exit.


All of the above information are generated in real time (including the loss function plot), so you find the experiment performaning really bad, you can exit early to save time.


By saving the entire training process, we can resume training from any checkpoint and adjust the learning rate if necessary.


## Training

Use [train_template.py](./train_template.py) as the entry point for training. 

There are 6 steps you need to do before start training
1. Follow [Dataset Setup Instructions](../datasets/README.md) to setup the training datasets.
2. Import the model you want to train with
   e.g. `from model.UNetSR import UNetNoTop, UNetD4, UNetSR`
3. Within [train_template.py](./train_template.py), set **RESULT_PATH** to the path of output, checkpoints will be saved there
4. Define models
    ```python
    unetsr = UNetSR(in_c=3, out_c=3, output_paddings=[1, 1]).to(device)
    srcnn = SRCNN().to(device)
    ```
5. Create configuration dictionaries (go through each parameter carefully, do not make a mistake here, or lots of time will be wasted).
   See [Training Configuration](#training-configuration) for what each parameter stands for
   Be careful of the **optimizer** parameter which you need to put your defined model here. Don't forget to update this after copy pasting the configuration dictionary. 
6. Add the models and configurations to the lists at the bottom of the template in corresponding order, **SR** supports training multiple models sequentially so that you don't have to manually start every experiment. 

### Training Configuration
- **epochs**
  Total number of epochs to train
- **save_period**
  The period to save a checkpoint (validation result and weight file)
- **batch_size**
  batch_size for batch training
- **checkpoint_dir**
  Output directory for saving checkpoint data and log file.
- **start_epoch**
  The epoch to start at. Default should be 1 for a new experiment. For resume-training, set **start_epoch** to be a epoch that you already have a weight file of. Then **SR** will load the weights and start training from epoch **(start_epoch + 1)**
- **criterion**
  Criterion (loss function). Can be `nn.MSELoss()`. Or use whatever criterion that is suitable for super resolution problem such as perceptual loss.
  ```python
  fe = FeatureExtractor().to(device)
  criterion = LossSum(fe)
  ```
    
- **dataset**
  Dataset path, can be either (already imported within the template)
  - TEXT_DATASET_PATH
  - DIV2K_DATASET_PATH
- **dataset_type**
  - size category for a dataset check [datasets](../datasets/README.md) for more info.
  - Most of the time, we choose 'same_300' for **DIV2K_DATASET_PATH** and 'same' for **TEXT_DATASET_PATH**
- **low_res**
  - Low quality input image type for a dataset
  - a number for **DIV2K_DATASET_PATH**, such as **100**
  - a category for **TEXT_DATASET_PATH**, such as **Blurradius3**, **Blurradius5**, **Blurradius7**
  - check the 
- **high_res**
  - High quality target image type for a dataset
  - a number for **DIV2K_DATASET_PATH**, such as **300**
  - a category for **TEXT_DATASET_PATH**, always **Target300x300**
- **device**
  - No need to modify, it's been defined at the beginning of the training script
- **scheduler**
  ```python
  # example
  'scheduler': {
    'step_size': 5,
    'gamma': 0.9
  }
  ```
  Scheduler for learning rate decay. Set it to None if you don't want learning rate decay.
- **optimizer**
  ```python
  # example
  'optimizer': optim.Adam(model.parameters(), lr=0.002)
  # or
  'optimizer': optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  ```
  Don't forget to update the model when you have defined multiple models in the training script. Using a wrong model here could result in unexpected results.
- **train_set_percentage**
  - Percentage of training dataset that will be used for training, we pick **0.9**.
  - The rest **0.1** are for validation. And the validation dataset will actually be used as our testing dataset.
- **num_worker**
  - Number of CPU cores for DataLoader.
  - Default: `multiprocessing.cpu_count()`
  - If any error occurs related to multiprocessing, try change this parameter to 0.
- **test_all_multiprocess_cpu**
  The number of CPU cores to use for generating test images. If error occurs or got stuck, it may be due to race condition (happens from time to time). To fix it, change it to 1.
- **test_only**
  - `False` for training, test images will be generated after training is done.
  - `True` for testing only, training process is skipped and **SR** will pick the last weight file from **checkpoint_dir** and run test images.

### Resume Training

There are only 2 parameters to change from configuration dictionary

suppose you have trained for 50 epoch and decide that you want to train for an extra 30 epochs.

Change these 2 parameters.

```python
'epochs': 80,
'start_epoch': 50
```

Keep the **checkpoint_dir** unchanged, **checkpoint_dir** has to contain weight file for the new **start_epoch**, in this case '**epoch50.pth** should exist in **checkpoint_dir**.

Optionally, you can change **scheduler**. If **scheduler** is changed to `None`, weight decay will stop from where you resume training. The learning rate will be the learning rate of your last checkpoint as opmizer state is also saved and loaded.

## Testing

