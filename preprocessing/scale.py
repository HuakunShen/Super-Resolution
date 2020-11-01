import cv2
import os
import argparse
from enum import Enum


class Mode(Enum):
    UP = 'up'
    DOWN = 'down'


parser = argparse.ArgumentParser("Image Scaler")
parser.add_argument('-f', '--factor', type=float, required=True, help='scaling factor')
parser.add_argument('-m', '--mode', type=Mode, choices=Mode, help='scale mode, choose from "up" or "down"')
parser.add_argument('-i', '--input', type=str,
                    help='input path/directory. If a directory is passed in, all images in it will be processed')
parser.add_argument('-o', '--output', type=str,
                    help='output path, if input is a directory, output must also be a directory')
parser.add_argument('-e', '--ext',
                    help='file extension, for example, your input is a directory and you only want to '
                         'scale all the .png files')
parser.add_argument('--interpolate', action='store_true',
                    help='Use this option to indicate whether you want to use interpolation for scaling')


def scale(image, factor: float, down=True, interpolation=True):
    print("down" if down else "up")
    if down:
        factor = 1.0 / factor
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    if interpolation:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, (width, height))
    return resized


if __name__ == '__main__':
    args = parser.parse_args()
    for key, value in args.__dict__.items():
        print("{}: {}".format(key, value))
    # validate arguments
    if os.path.isfile(args.input):
        if os.path.exists(args.output):
            if os.path.isdir(args.output):
                raise ValueError(
                    "Invalid input. Input is a file, output is an existing directory. "
                    "Use another output or remove the output")
            ans = input("output file {} already exists, want to remove? Y/n: ".format(args.output))
            if ans.lower() == 'y':
                os.remove(args.output)
            else:
                exit(0)
        else:
            print("output is valid")
    elif os.path.isdir(args.input):
        if os.path.exists(args.output):
            assert os.path.isdir(args.output)
    else:
        raise ValueError("Unknown Error")

    if os.path.isfile(args.input):
        img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        print('Original Dimensions : ', img.shape)
        output_img = scale(img, args.factor, args.mode == Mode.DOWN, interpolation=args.interpolate)
        print('Resized Dimensions : ', output_img.shape)
        cv2.imwrite(args.output, output_img)
    elif os.path.isdir(args.input):
        all_files = os.listdir(args.input)
        if args.ext:
            all_files = list(filter(lambda f: f.endswith(args.ext), all_files))
        output_dir = os.path.abspath(args.output)
        if not os.path.exists(output_dir):
            print("output dir DNE, creating it")
            os.makedirs(output_dir)
        for file in all_files:
            file_path = os.path.join(args.input, file)
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Read img is none, filename: {}".format(file_path))
            print("file: {}".format(file))
            print('Original Dimensions : ', img.shape)
            output_img = scale(img, args.factor, args.mode == Mode.DOWN, interpolation=args.interpolate)
            print('Resized Dimensions : ', output_img.shape)
            output_path = os.path.join(os.path.abspath(args.output), file)
            print("output path: {}".format(output_path))
            cv2.imwrite(output_path, output_img)
    else:
        print("Unknown Error")
