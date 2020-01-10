import sources.DQN as DQN
import torch
import torch.nn as nn
import gym


class Network(nn.Module):
    def __init__(self, output_shape):
        super(Network, self).__init__()
        self.conv = nn.Sequential(     # input_shape: 3*210*160
            nn.Conv2d(
                in_channels=3,     # RGB: 3 channels
                out_channels=16,   # filters' number
                kernel_size=5,     # kernel's size
                stride=2,
                padding=2
            ),                          # output_shape: 16*105*80
            nn.ReLU(),
            nn.MaxPool2d(               # input_shape: 16*105*80
                kernel_size=3,
                stride=1,
                padding=1,
            ),                          # output_shape: 16*105*80
            nn.Conv2d(                  # input_shape: 16*105*80
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),                          # output_shape: 32*53*40
            nn.ReLU(),
            nn.MaxPool2d(  # input_shape: 32*53*40
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # output_shape: 32*53*40
            nn.Conv2d(  # input_shape: 32*53*40
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2
            ),  # output_shape: 64*27*20
            nn.ReLU(),
            nn.Conv2d(  # input_shape: 64*27*20
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),  # output_shape: 64*14*10
            nn.ReLU()
        )
        self.fcn = nn.Sequential(
            nn.Linear(64*14*10, 64*14*2),
            nn.ReLU(),
            nn.Linear(64*14*2, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape),
        )

    def forward(self, state):
        out = self.conv(state)
        out = out.view(out.size(0), -1)
        out = self.fcn(out)
        return out


if __name__ == '__main__':
    env = gym.make("SpaceInvaders-v0")
    observation = env.reset()
    observation = torch.FloatTensor(observation)
    observation = torch.transpose(observation, 0, 2)
    observation = torch.transpose(observation, 1, 2).unsqueeze(0)

    net = Network(env.action_space.n)
    out = net.forward(observation)
