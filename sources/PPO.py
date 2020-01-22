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
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        if self.discrete:
            self.policy = nn.Sequential(
                nn.Linear(n_state, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_action)
            )
        else:
            self.policy = nn.Sequential(
                nn.Linear(n_state, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh()
            )
            self.mean = nn.Linear(64, n_action)
            self.logstd = nn.Linear(64, n_action)

    def forward(self, state):
        action_values = self.state_value(state)
        actions = self.policy(state)
        if self.discrete:
            action_prob = torch.nn.functional.softmax(actions, dim=-1)
            # print('action_prob: ', action_prob)
            distribution = Categorical(action_prob)
            return distribution, action_values
        else:
            mean, logstd = self.mean(actions), self.logstd(actions)
            std = torch.nn.functional.softplus(logstd)
            mean = torch.tanh(mean) * 2
            distribution = Normal(loc=mean, scale=std)
            return distribution, action_values


class Model:
    def __init__(self, net, device, n_state, n_action, discrete, learn=False, gamma=0.95, learning_rate=0.005,
                 epsilon=0.1):
        # init networks
        if learn is True:
            # self.net = net(output_shape=n_action).to(device)
            self.net = net(n_state, n_action, discrete).to(device)
        else:
            self.load_net()
        # self.old_net = net(output_shape=n_action).to(device)
        self.old_net = net(n_state, n_action, discrete).to(device)
        self.n_action = n_action
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.device = device

    def learn(self, state_collections, action_collections, advantage_collections, value_target):
        """
        :param state_collections: Tensor[T, state_features]
        :param action_collections: Tensor[T] or Tensor[T, action_dimensions]
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

        self.old_net.load_state_dict(self.net.state_dict())
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.learning_rate)
        idx_list = self.batch(T, 2)
        for j in range(4):
            # loss = mean(ratio * advantages) - lambda * KL(old_net, net)
            # J = -loss
            # print(torch.exp(self.policy.forward(state_collections).log_prob(action_collections)))
            for idx in idx_list:
                idx = torch.LongTensor(idx).to(device=self.device)
                state_sample = state_collections.index_select(0, idx)
                action_sample = action_collections.index_select(0, idx)
                advantage_sample = advantage_collections.index_select(0, idx)
                predict_sample = predict_value.index_select(0, idx)
                target_sample = value_target.index_select(0, idx)
                if self.net.discrete is True:
                    ratio = torch.div(torch.exp(self.net.forward(state_sample)[0].log_prob(action_sample)),
                                      torch.exp(self.old_net.forward(state_sample)[0].log_prob(action_sample)).detach())
                else:
                    # logp = self.net.forward(state_sample)[0].log_prob(action_sample).squeeze()
                    # old_logp = self.old_net.forward(state_sample)[0].log_prob(action_sample).squeeze()
                    dis = self.net.forward(state_sample)[0]
                    mean, std = dis.mean, dis.stddev
                    old_dis = self.old_net.forward(state_sample)[0]
                    old_mean, old_std = old_dis.mean, old_dis.stddev
                    logp = -(0.5 * torch.pow((action_sample - mean) / std, 2) + torch.log(std))
                    old_logp = -(0.5 * torch.pow((action_sample - old_mean) / old_std, 2) + torch.log(old_std))
                    ratio = torch.div(torch.exp(logp), torch.exp(old_logp)).squeeze()
                # a = torch.mul(ratio, advantage_sample.detach())
                clip_loss = - torch.mean(torch.min(torch.mul(ratio, advantage_sample.detach()),
                                                   torch.mul(torch.clamp(ratio, min=1-self.epsilon, max=1+self.epsilon),
                                                             advantage_sample.detach())
                                                   )
                                         )
                vf_loss = torch.mean(torch.pow(predict_sample - target_sample, 2))
                loss = vf_loss + clip_loss

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

    def choose_action(self, state):
        # print(self.policy.forward(state).probs)
        a = self.net.forward(state)[0]
        action = self.net.forward(state)[0].sample()
        return action

    def batch(self, T, batch_size):
        idx = [i for i in range(T)]
        idx_list = []
        temp = []
        for i in range(batch_size):
            for j in random.sample(idx, T // batch_size):
                temp.append(j)
                idx.remove(j)
            idx_list.append(temp)
            temp = []
        return idx_list

    def save_net(self):
        torch.save(self.net, 'params/ppo_net.pkl')

    def load_net(self):
        self.net = torch.load('params/ppo_net.pkl')
