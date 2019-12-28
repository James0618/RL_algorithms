import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical
import torch.multiprocessing as mp


class ValueNet(nn.Module):
    def __init__(self, n_state):
        super(ValueNet, self).__init__()
        self.state_value = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        action_values = self.state_value(state)
        return action_values


class PolicyNet(nn.Module):
    def __init__(self, n_state, n_action):
        super(PolicyNet, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_action),
        )

    def forward(self, state):
        actions = self.policy(state)
        action_prob = torch.nn.functional.softmax(actions, dim=-1)
        # print('action_prob: ', action_prob)
        distribution = Categorical(action_prob)
        return distribution


class Model:
    def __init__(self, n_state, n_action, learn=False, gamma=0.95, learning_rate=0.005, epsilon=0.1):
        # init networks
        if learn is True:
            self.policy = PolicyNet(n_state=n_state, n_action=n_action)
            self.state_value = ValueNet(n_state=n_state)
        else:
            self.load_net()
        self.old_policy = PolicyNet(n_state=n_state, n_action=n_action)

        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def learn(self, state_collections, action_collections, reward_collections):
        """
        :param state_collections: Tensor[T, state_features]
        :param action_collections: Tensor[T]
        :param reward_collections: Tensor[T]
        :return:
        """
        T = state_collections.shape[0]
        returns = torch.FloatTensor([0])  # V(S_final)
        advantage_collections = torch.FloatTensor([])  # Tensor[T]
        for i in range(1, T + 1):
            # t: T -> 1
            # Returns[t] = r[t] + gamma * Returns[t+1]
            returns = reward_collections[T - i] + self.gamma * returns
            advantage = returns - self.state_value.forward(state_collections[T - i])
            if advantage_collections.shape[0] == 0:
                advantage_collections = advantage.unsqueeze(0)
            else:
                advantage_collections = torch.cat((advantage_collections, advantage.unsqueeze(0)))

        idx = [i for i in range(advantage_collections.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        advantage_collections = advantage_collections.index_select(0, idx)

        torch.save(self.policy.state_dict(), 'params/old_policy_net.pkl')
        old_policy = PolicyNet(n_state=self.n_state, n_action=self.n_action)
        old_policy.load_state_dict(torch.load('params/old_policy_net.pkl'))

        policy_optimizer = torch.optim.SGD(params=self.policy.parameters(), lr=self.learning_rate)
        state_value_optimizer = torch.optim.SGD(params=self.state_value.parameters(), lr=2*self.learning_rate)
        # loss_func = nn.KLDivLoss()
        for j in range(12):
            # loss = mean(ratio * advantages) - lambda * KL(old_net, net)
            # J = -loss
            # print(torch.exp(self.policy.forward(state_collections).log_prob(action_collections)))
            ratio = torch.div(torch.exp(self.policy.forward(state_collections).log_prob(action_collections)),
                              torch.exp(old_policy.forward(state_collections).log_prob(action_collections)))
            policy_loss = - torch.mean(torch.min(torch.mul(ratio, advantage_collections.detach()),
                                                 torch.mul(torch.clamp(ratio, min=1-self.epsilon, max=1+self.epsilon),
                                                           advantage_collections.detach())
                                                 )
                                       )
            # print(policy_loss)
            # policy_loss = -(-self.lamb * loss_func(torch.log(old_policy.forward(state_collections).probs),
            #                                            torch.log(self.policy.forward(state_collections).probs)) +
            #                     torch.mean(torch.mul(
            #                         torch.exp(self.policy.forward(state_collections).log_prob(action_collections) -
            #                                   old_policy.forward(state_collections).log_prob(action_collections)),
            #                         advantage_collections.detach())))
            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

        for j in range(16):
            returns = torch.FloatTensor([0])
            advantage_collections = torch.FloatTensor([])  # Tensor[T]
            for i in range(1, T + 1):
                # t: T -> 1
                # Returns[t] = r[t] + gamma * Returns[t+1]
                returns = reward_collections[T - i] + self.gamma * returns
                advantage = returns - self.state_value.forward(state_collections[T - i])
                if advantage_collections.shape[0] == 0:
                    advantage_collections = advantage.unsqueeze(0)
                else:
                    advantage_collections = torch.cat((advantage_collections, advantage.unsqueeze(0)))
            state_loss = torch.mean(torch.pow(advantage_collections, 2))
            state_value_optimizer.zero_grad()
            state_loss.backward(retain_graph=True)
            state_value_optimizer.step()

    def choose_action(self, state):
        # print(self.policy.forward(state).probs)
        action = self.policy.forward(state).sample()
        return action

    def save_net(self):
        torch.save(self.policy, 'params/ppo_policy.pkl')
        torch.save(self.state_value, 'params/dppo_state_value.pkl')

    def load_net(self):
        self.policy = torch.load('params/ppo_policy.pkl')
        self.state_value = torch.load('params/dppo_state_value.pkl')

class Worker(mp.Process)