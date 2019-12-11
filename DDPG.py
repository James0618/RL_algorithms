import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import random

class DQN(nn.Module):
    def __init__(self, n_replay, n_state, n_action, learning_rate, gamma):
        super(DQN, self).__init__()

        # model init
        self.q_net = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_action),
            nn.ReLU()
        )
        self.optimizer = torch.optim.SGD(self.q_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        # parameters init
        self.n_replay = n_replay
        self.n_state = n_state
        self.n_action = n_action
        self.lr = learning_rate
        self.gamma = gamma
        self.batch_size = 16

        # replay init
        self.replay = []
        self.full = False
        self.pointer = 0

    def forward(self, state):
        return self.q_net(state)

    def train_one_step(self):
        if len(self.replay) < self.batch_size:
            transitions = self.replay
        else:
            transitions = random.sample(self.replay, self.batch_size)
        data_array = np.array(transitions)
        data_state = data_array[:, 0:self.n_state]
        data_state_ = data_array[:, -self.n_state:]
        # TODO: set zero array and replay current action value

    # replay space
    def store_transition(self, transition):
        """
        :param transition: list - state, action, reward, state'
        :return: None
        """
        if self.pointer < self.n_replay:
            if not self.full:
                self.replay.append(transition)
            else:
                self.replay[self.pointer] = transition
            self.pointer += 1
        else:
            self.pointer = 0
            self.replay[self.pointer] = transition



if __name__ == '__main__':
    agent = DQN(1000, 3, 3)
