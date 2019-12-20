import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical


class ValueNet(nn.Module):
    def __init__(self, n_state):
        super(ValueNet, self).__init__()
        self.state_value = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state):
        action_values = self.state_value(state)
        return action_values


class PolicyNet(nn.Module):
    def __init__(self, n_state, n_action):
        super(PolicyNet, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_action),
        )

    def forward(self, state):
        actions = self.policy(state)
        action_prob = torch.nn.functional.softmax(actions)
        distribution = Categorical(action_prob)
        return distribution


class Model:
    def __init__(self, n_state, n_action, gamma=0.95, lamb=0.5, kl_target=0.01, learning_rate=0.005):
        # init networks
        self.policy = PolicyNet(n_state=n_state, n_action=n_action)
        self.old_policy = PolicyNet(n_state=n_state, n_action=n_action)
        self.state_value = ValueNet(n_state=n_state)

        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma
        self.lamb = lamb
        self.kl_target = kl_target
        self.learning_rate = learning_rate

    def learn(self, policy, state_value, state_collections, action_collections, reward_collections, final_state):
        """
        :param policy: policy net
        :param state_value: state value net
        :param state_collections: Tensor[T, state_features]
        :param action_collections: Tensor[T, action_numbers]
        :param reward_collections: Tensor[T, 1]
        :param final_state: Tensor[state_features]
        :return:
        """
        T = state_collections.shape[0]
        returns = self.state_value(final_state)  # V(S_final)
        advantage_collections = []
        for i in range(1, T + 1):
            # t: T -> 1
            # Returns[t] = r[t] + gamma * Returns[t+1]
            returns = reward_collections[T - i] + self.gamma * returns
            advantage = returns - self.state_value(state_collections[T - i])
            advantage_collections.append(advantage)
        advantage_collections = torch.FloatTensor(advantage_collections)

        torch.save(policy.state_dict(), 'params/old_policy_net.pkl')
        old_policy = PolicyNet(n_state=self.n_state, n_action=self.n_action)
        old_policy.load_state_dict(torch.load('params/old_policy_net.pkl'))

        policy_optimizer = torch.optim.SGD(params=policy.parameters(), lr=self.learning_rate)
        state_value_optimizer = torch.optim.SGD(params=state_value.parameters(), lr=self.learning_rate)
        loss_func = nn.KLDivLoss()
        for j in range(10):
            # loss = mean(ratio * advantages) - lambda * KL(old_net, net)
            # J = -loss
            loss = - (- self.lamb * loss_func(old_policy(state_collections), policy(state_collections)) +
                      torch.mean(torch.mul(torch.exp(policy(state_collections).log_prob(action_collections) -
                                                     old_policy(state_collections).log_prob(action_collections)),
                                           advantage_collections.detach())))
            policy_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()

        for j in range(10):
            loss = torch.mean(torch.pow(advantage_collections, 2))
            state_value_optimizer.zero_grad()
            loss.backward()
            state_value_optimizer.step()

        if loss_func(old_policy(state_collections), policy(state_collections)) > self.kl_target * 1.5:
            self.lamb = self.lamb * 2
        elif loss_func(old_policy(state_collections), policy(state_collections)) < self.kl_target / 1.5:
            self.lamb = self.lamb / 2

    def choose_action(self, state):
        action = self.policy.forward(state).sample()
        return action


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    model = Model(n_state=2*env.observation_space.shape[0], n_action=env.action_space.n, learning_rate=0.005)
    env.reset()

    for episode in range(5000):
        observation = env.reset()
        state = observation
        reward = 0
        for t in range(1000):
            if episode > 4000:
                env.render()
            if t < 1:
                observation, _, done, info = env.step(0)
                state = np.append(state, observation)
            else:
                state_before = state
                action = model.choose_action(state)
                # print('action: {}'.format(action))
                observation, _, done, info = env.step(int(action))

                x, x_dot, theta, theta_dot = observation
                # use the reward as Morvan Zhou defined
                # https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                reward = r1 + r2
                state = np.append(state[-4:], observation)
                if done:
                    print("Episode {}: finished after {} timesteps".format(episode, t+1))
                    break
        print("Episode {}: finished after {} timesteps".format(episode, 'max'))

    env.close()