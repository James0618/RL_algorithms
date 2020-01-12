import torch
import torch.nn as nn
import numpy as np
import random
import gym


class DQN:
    def __init__(self, network, device, n_replay, n_action, learn=True, learning_rate=0.005, gamma=0.95, epsilon=0.1):
        # parameters init
        self.n_replay = n_replay
        self.n_action = n_action
        self.device = device

        if learn is False:
            self.load_net('dqn-atari')
        else:
            self.q_net = network

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.times = 0
        self.batch_size = 32

        # replay init
        self.replay = []
        self.full = False
        self.pointer = 0

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).to(device=self.device)
        self.times += 1
        if self.times < 1e6:
            epsilon = (((self.epsilon - 1) * self.times) / 1e6) + 1
        else:
            epsilon = self.epsilon
        # epsilon-argmax
        if random.random() < epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = self.q_net.forward(state_tensor).max(1)[1]
            # print(action)
        return int(action)

    def learn(self):
        # check if replay is available
        if len(self.replay) == 0:
            raise ValueError('Replay is empty!')

        if len(self.replay) < self.batch_size:
            transitions = self.replay
        else:
            transitions = random.sample(self.replay, self.batch_size)

        # extract data in mini-batch
        # transition: state -> action -> reward -> state'
        #             state & state' - FloatTensor
        #             action - int
        #             reward - float
        # data_state & data_state_ - [N_Batch, state_size]
        # action & reward -[N_Batch, 1]
        for index, transition in enumerate(transitions):
            if index == 0:
                data_state = transition[0]
                data_action = torch.LongTensor([transition[1]]).unsqueeze(0)
                data_reward = torch.FloatTensor([transition[2]]).unsqueeze(0)
                data_state_ = transition[3]
            else:
                data_state = torch.cat((data_state, transition[0]), dim=0)
                data_action = torch.cat((data_action, torch.LongTensor([transition[1]]).unsqueeze(0)), dim=0)
                data_reward = torch.cat((data_reward, torch.FloatTensor([transition[2]]).unsqueeze(0)), dim=0)
                data_state_ = torch.cat((data_state_, transition[3]), dim=0)

        data_state = data_state.to(device=self.device)
        data_action = data_action.to(device=self.device)
        data_reward = data_reward.to(device=self.device)
        data_state_ = data_state_.to(device=self.device)

        # calculate q_current and q_next
        q_value = self.q_net.forward(data_state).gather(1, data_action)
        q_next = self.q_net.forward(data_state_).max(1)[0].view(len(transitions), 1)
        target = data_reward + self.gamma * q_next
        # print('q_value, q_next, target: ', q_value, q_next, target)
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
        # print(transition)
        if self.pointer < self.n_replay:
            if not self.full:
                self.replay.append(transition)
            else:
                self.replay[self.pointer] = transition
            self.pointer += 1
        else:
            self.full = True
            self.pointer = 0
            self.replay[self.pointer] = transition

    def save_net(self, name):
        torch.save(self.q_net, 'params/{}.pkl'.format(name))

    def load_net(self, name):
        self.q_net = torch.load('params/{}.pkl'.format(name))
