import os
import pathlib

SR_PATH = pathlib.Path(__file__).parent.absolute()
DIV2K_DATASET_PATH = (SR_PATH.parent / 'datasets' / 'DIV2K').absolute()
TEXT_DATASET_PATH = (SR_PATH.parent / 'datasets' / 'TEXT').absolute()
MSG_DIVIDER_LEN = 100
