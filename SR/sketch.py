import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':

    images = torch.rand((4, 600, 600, 3))
    fig, axes = plt.subplots(nrows=len(images), ncols=3, figsize=(50, 50))
    for row in range(len(images)):
        for col in range(3):
            axes[row][col].set_title('hello')
            axes[row][col].axis('off')
            axes[row][col].imshow(images[row])

    fig.savefig('/home/hacker/Desktop/test_image.png')
    plt.close()
