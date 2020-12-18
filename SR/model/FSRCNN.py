"""
@author: Huakun Shen, yjn870
@reference: https://github.com/yjn870/FSRCNN-pytorch/blob/master/models.py
"""
import math
from torch import nn
import torch


class FSRCNN_Original(nn.Module):
    """
    This class is copied from https://github.com/yjn870/FSRCNN-pytorch/blob/master/models.py
    fsrcnn_original = FSRCNN_Original(scale_factor=3, num_channels=3).to(device)
    cnn_original_config = {
        'epochs': 20,
        'save_period': 5,
        'batch_size': 16,
        'checkpoint_dir': SR_path/'result/FSRCNN-50-150-original',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'DATASET_TYPE': 'diff',
        'low_res': 100,
        'high_res': 300,
        'learning_rate': 1e-3
    }
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': config['learning_rate'] * 0.1}
    ], lr=config['learning_rate'])
    """

    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN_Original, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend(
                [nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9 // 2,
                                            output_padding=scale_factor - 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(
                    2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(
                    2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


class FSRCNN(nn.Module):
    """
    fsrcnn_config = {
        'epochs': 50,
        'save_period': 10,
        'batch_size': 16,
        'checkpoint_dir': SR_path/'result/FSRCNN-100-300',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': nn.MSELoss(),
        'DATASET_TYPE': 'diff',
        'low_res': 100,
        'high_res': 300,
        'learning_rate': 0.001
    }
    """

    def __init__(self, factor: int) -> None:
        super(FSRCNN, self).__init__()
        tmp = []
        for _ in range(4):
            tmp.append(nn.Conv2d(16, 16, kernel_size=3, padding=1))
            tmp.append(nn.PReLU(16))

        self.body = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.PReLU(64),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.PReLU(16),
            *tmp,
            nn.Conv2d(16, 64, kernel_size=1),
            nn.PReLU(64),
            nn.ConvTranspose2d(64, 3, kernel_size=9, stride=factor, padding=4,
                               output_padding=factor - 1)
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.body:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias.data)

    def forward(self, inputs):
        return self.body(inputs)


if __name__ == "__main__":
    model = FSRCNN(3)
    img = torch.rand((1, 3, 150, 150))
    print(model(img).shape)
