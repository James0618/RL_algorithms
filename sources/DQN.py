import torch
import torch.nn as nn
import numpy as np
import random
import gym


class DQN:
    def __init__(self, network, n_replay, n_state, n_action, load_param=False, learning_rate=0.005, gamma=0.95,
                 epsilon=0.1):
        # parameters init
        self.n_replay = n_replay
        self.n_state = n_state
        self.n_action = n_action

        if load_param is True:
            self.load_net()
        else:
            self.q_net = network(n_state=self.n_state, n_action=self.n_action)

        self.optimizer = torch.optim.SGD(self.q_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = 64

        # replay init
        self.replay = []
        self.full = False
        self.pointer = 0

    def choose_action(self, state):
        state_tensor = torch.FloatTensor([state])
        # epsilon-argmax
        if random.random() < self.epsilon:
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
                data_state = transition[0].unsqueeze(0)
                data_action = transition[1].unsqueeze(0)
                data_reward = transition[2].unsqueeze(0)
                data_state_ = transition[3].unsqueeze(0)
            else:
                data_state = torch.cat((data_state, transition[0].unsqueeze(0)), dim=0)
                data_action = torch.cat((data_action, transition[1].unsqueeze(0)), dim=0)
                data_reward = torch.cat((data_reward, transition[1].unsqueeze(0)), dim=0)
                data_state_ = torch.cat((data_state_, transition[0].unsqueeze(0)), dim=0)

        # calculate q_current and q_next
        q_value = self.q_net.forward(data_state).gather(1, data_action)
        q_next = self.q_net.forward(data_state_).max(1)[0].view(len(transitions), 1)
        target = data_reward + self.gamma * q_next
        # print('q_value, q_next, target: ', q_value, q_next, target)
        loss = self.loss_func(input=q_value, target=target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print('###################################')
        # for name, param in self.q_net.named_parameters():
        #     if param.requires_grad:
        #         print(param)

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
            self.pointer = 0
            self.replay[self.pointer] = transition

    def save_net(self):
        torch.save(self.q_net, 'params/dqn.pkl')

    def load_net(self):
        self.q_net = torch.load('params/dqn.pkl')
