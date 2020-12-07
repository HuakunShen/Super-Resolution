# Super Resolution

## Dataset

https://data.vision.ee.ethz.ch/cvl/DIV2K/

## Data

A python script `./preprocessing/scale.py` is made for scaling up or down images.

It supports single file scaling or batch scaling

```
python scale.py --help
usage: Image Scaler [-h] -f FACTOR [-m {Mode.UP,Mode.DOWN}] [-i INPUT] [-o OUTPUT] [-e EXT] [--interpolate]

optional arguments:
  -h, --help            show this help message and exit
  -f FACTOR, --factor FACTOR
                        scaling factor
  -m {Mode.UP,Mode.DOWN}, --mode {Mode.UP,Mode.DOWN}
                        scale mode, choose from "up" or "down"
  -i INPUT, --input INPUT
                        input path/directory. If a directory is passed in, all images in it will be processed
  -o OUTPUT, --output OUTPUT
                        output path, if input is a directory, output must also be a directory
  -e EXT, --ext EXT     file extension, for example, your input is a directory and you only want to scale all the .png files
  --interpolate         Use this option to indicate whether you want to use interpolation for scaling
```

```bash
# single file scaling
python scale.py --mode down -i 0801.png -o 0801down.png --factor 10
# batch scaling
python scale.py --mode down -i D:\Downloads\DIV2K_valid_HR -o resized -f 20

# add --interpolate if you want to scale with interpolation (smoother)
```

