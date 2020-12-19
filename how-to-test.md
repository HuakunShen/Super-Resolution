# How to test?

We prepared 2 types of testing
- Single Image Testing
- Batch Testing

Both methods require some data preparation.

For both methods, most of our model supports 300x300 input and 300x300 output. Images of other sizes may or may not work for some models as some models have some restrictions on sizes.

## Prepare a Weight File

The model weights are from our training experiments. All the experiments results can be found [here](https://onedrive.live.com/?authkey=%21AFA6I61NVeysBP8&id=7A78FD2CB5D891D5%21161697&cid=7A78FD2CB5D891D5).

See [SR Training Framework](./SR) for more information about how it works and what data are generated from an experiments.

Basically, an experiment contains:
- **loss.png** (or train_loss.png and valid_loss.png depending on the version of framework we used)
  - Training Loss and Validation Loss function
- **validation**
  - A directory containing validation results saved periodically during the training process
  - See the progress of how the model is trained and from the last validation image, you can see the final performance of the experiement.
- **weights**
  - A directory containing pytorch weights files that are saved periodically during training process.
- **test** (there may or may not be a test folder)
  - A directory containing all the test results.
  - There may or may not be a test folder depending on the framework version. If there is not, check the last validation image from the **validation** folder
- **log.log** (some old experiments may not have this)
  - From the experiment folder, you can have a brief idea of what model is used. From **log.log** you can find the exact training configuration including learning rate, optimizer, scheduler, dataset used, and other information such as Peak GPU memory usage, Peak Memory Usage, error messages. Continuous training records.

From the above information, you can pick a good experiment and a weight file from it for testing.

## Testing Method 1

Batch Testing is enabled by default during the training process. Once training is done, a test directory is created and all test result are saved to it. See [SR Training Framework](./SR) for details.

You can use the same file for training to do the testing.

[train_template.py](./SR/train_template.py) contains the code for configuration.

```python
config = {
        ...
        'checkpoint_dir': '<the experiment checkpoint directory>',
        'test_only': True
        ...
    }
```

To do testing, simply change **test_only** parameter in the configuration dictionary to True, and it will skip all training and do the testing with the last weight file in the checkpoint directory.

Since Batch Testing uses our dataset images, you have to setup the dataset properly. See [Dataset Setup Instruction](./datasets) for instructions.

Note: 
- **checkpoint_dir** has to be set to an existing experiment directory with weights. Suggestion: Download your chosen experiment from our cloud drive.
- A model needs to be defined just like the training setup (e.g. `model=`), see [model](./SR/model/README.md) for all model definitions.

## Testing Method 2 (recommended)

[Video Demo](https://youtu.be/oldS47apL7s)

Use the [test.py](./SR/test.py) for testing. 

This is a python script that takes in images and a weight file, then output the computed images.

This script supports 3 modes:
- single image testing
- batch testing without target image
- batch testing with target image

```python
# Example for batch testing with target images
python test.py --weight model.pth --output test --batch_size 10 --input valid_100 --label valid_300

# Example for batch testing without target images
python test.py --weight model.pth --output test --batch_size 10 --input valid_100

# Example for single image testing
python test.py --weight model.pth --output output_file.png --input input_file.png
```

[test.py](./SR/test.py) is in [SR](./SR) directory.

For the arguments to the python script, use the correct path.

