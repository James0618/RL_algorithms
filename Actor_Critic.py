import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    def __init__(self, n_state, n_action):
        super(Network, self).__init__()

        self.q_net = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_action),
        )

    def forward(self, state):
        action_values = self.q_net(state)
        return action_values


class ActorCritic:
    def __init__(self, n_state, n_action, load_param=False, learning_rate=0.1, gamma=0.95, epsilon=0.1):
        # parameters init
        self.n_state = n_state
        self.n_action = n_action
