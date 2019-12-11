import torch
import torch.nn as nn
import numpy as np
import random


class DQN(nn.Module):
    def __init__(self, n_replay, n_state, n_action, learning_rate=0.05, gamma=0.9, epsilon=0.1):
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
        self.epsilon = epsilon
        self.batch_size = 16

        # replay init
        self.replay = []
        self.full = False
        self.pointer = 0

    def forward(self, state):
        return self.q_net(state)

    def choose_action(self, state):
        # epsilon-argmax
        if random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = self.q_net(state).max(1)[0].numpy()
        return action

    def learn(self):
        # check if replay is available
        if len(self.replay) == 0:
            raise ValueError('Replay is empty!')

        if len(self.replay) < self.batch_size:
            transitions = self.replay
        else:
            transitions = random.sample(self.replay, self.batch_size)

        # extract data in mini-batch
        data_array = np.array(transitions)
        data_state = torch.from_numpy(data_array[:, 0:self.n_state]).float()
        data_action = torch.from_numpy(data_array[:, self.n_state:self.n_state+1]).long()
        data_reward = torch.from_numpy(data_array[:, self.n_state+1:self.n_state+2]).float()
        data_state_ = torch.from_numpy(data_array[:, -self.n_state:]).float()

        # calculate q_current and q_next
        q_value = self.q_net(data_state).gather(1, data_action)
        q_next = self.q_net(data_state_).max(1)[0].view(len(transitions), 1)
        target = data_reward + self.gamma * q_next
        loss = self.loss_func(input=q_value, target=target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # replay space
    def store_transition(self, transition):
        """
        :param transition: list - state, action, reward, state'
        :return: None
        """
        # circular pointer
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
