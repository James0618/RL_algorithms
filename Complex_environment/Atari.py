import sources.DQN as DQN
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(     # input_shape: 3*210*160
            nn.Conv2d(
                in_channels=3,     # RGB: 3 channels
                out_channels=16,   # filters' number
                kernel_size=3,     # kernel's size
                stride=1,
                padding=1
            ),                          # output_shape: 3*210*160
            nn.ReLU()
        )
