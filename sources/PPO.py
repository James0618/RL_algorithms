import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical, Normal
import math
import random


class Net(nn.Module):
    def __init__(self, n_state, n_action, discrete=True):
        super(Net, self).__init__()
        self.discrete = discrete
        self.state_value = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        if self.discrete:
            self.policy = nn.Sequential(
                nn.Linear(n_state, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, n_action),
            )
        else:
            self.policy = nn.Sequential(
                nn.Linear(n_state, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, 2 * n_action),
            )

    def forward(self, state):
        action_values = self.state_value(state)
        actions = self.policy(state)
        if self.discrete:
            action_prob = torch.nn.functional.softmax(actions, dim=-1)
            # print('action_prob: ', action_prob)
            distribution = Categorical(action_prob)
            return distribution, action_values
        else:
            print(actions)
            mean, sigma = actions.chunk(chunks=2, dim=0)
            distribution = Normal(loc=mean, scale=sigma)
            return distribution, action_values


class Model:
    def __init__(self, net, device, n_state, n_action, learn=False, gamma=0.95, learning_rate=0.005,
                 epsilon=0.1):
        # init networks
        if learn is True:
            # self.net = net(output_shape=n_action).to(device)
            self.net = net(n_state, n_action).to(device)
        else:
            self.load_net()
        # self.old_net = net(output_shape=n_action).to(device)
        self.old_net = net(n_state, n_action).to(device)
        self.n_action = n_action
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.device = device

    def learn(self, state_collections, action_collections, advantage_collections, value_target):
        """
        :param state_collections: Tensor[T, state_features]
        :param action_collections: Tensor[T]
        :param advantage_collections: Tensor[T]
        :param value_target: Tensor[T]
        :return:
        """
        T = state_collections.shape[0]
        state_collections = state_collections.to(self.device)
        action_collections = action_collections.to(self.device)
        advantage_collections = advantage_collections.to(self.device)
        value_target = value_target.to(self.device)
        predict_value = self.net.forward(state_collections)[1]

        idx = [i for i in range(advantage_collections.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(device=self.device)
        advantage_collections = advantage_collections.index_select(0, idx)

        self.old_net.load_state_dict(self.net.state_dict())
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.learning_rate)
        for j in range(2):
            # loss = mean(ratio * advantages) - lambda * KL(old_net, net)
            # J = -loss
            # print(torch.exp(self.policy.forward(state_collections).log_prob(action_collections)))
            idx = random.sample(range(0, T), math.floor(T * 0.5) + 1)
            idx = torch.LongTensor(idx).to(device=self.device)
            state_sample = state_collections.index_select(0, idx)
            action_sample = action_collections.index_select(0, idx)
            advantage_sample = advantage_collections.index_select(0, idx)
            predict_sample = predict_value.index_select(0, idx)
            target_sample = value_target.index_select(0, idx)

            ratio = torch.div(torch.exp(self.net.forward(state_sample)[0].log_prob(action_sample)),
                              torch.exp(self.old_net.forward(state_sample)[0].log_prob(action_sample)))
            clip_loss = - torch.mean(torch.min(torch.mul(ratio, advantage_sample.detach()),
                                               torch.mul(torch.clamp(ratio, min=1-self.epsilon, max=1+self.epsilon),
                                                         advantage_sample.detach())
                                               )
                                     )
            vf_loss = torch.mean(torch.pow(predict_sample - target_sample, 2))

            loss = vf_loss + clip_loss

            optimizer.zero_grad()
            clip_loss.backward(retain_graph=True)
            optimizer.step()

            optimizer.zero_grad()
            vf_loss.backward(retain_graph=True)
            optimizer.step()

    def choose_action(self, state):
        # print(self.policy.forward(state).probs)
        action = self.net.forward(state)[0].sample()
        return action

    def save_net(self):
        torch.save(self.net, 'params/ppo_net.pkl')

    def load_net(self):
        self.net = torch.load('params/ppo_net.pkl')
