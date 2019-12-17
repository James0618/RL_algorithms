import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym


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


class Actor:
    def __init__(self, n_state, n_action, load_param=False, learning_rate=0.001, betas=tuple([0.9, 0.999]),
                 weighted_decay=0, gamma=0.95, epsilon=0.1):
        # parameters init
        self.n_state = n_state
        self.n_action = n_action

        self.policy = Network(n_state=n_state, n_action=n_action)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate,
                                          betas=betas, weight_decay=weighted_decay)

    def choose_action(self, state):
        policy = self.policy(torch.FloatTensor([state]))
        actions_distribution = Categorical(policy)
        action = actions_distribution.sample()
        return action

    def __loss_func(self, state, advantage, action):
        policy = self.policy(torch.FloatTensor([state]))
        actions_distribution = Categorical(policy)
        log_probability = actions_distribution.log_prob(action).unsqueeze(0)
        actor_loss = log_probability * advantage
        return actor_loss

    def learn(self, state, advantage, action):
        loss = self.__loss_func(state=state, advantage=advantage.detach(), action=action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Critic:
    def __init__(self, n_state, n_action, load_param=False, learning_rate=0.001, betas=tuple([0.9, 0.999]),
                 weighted_decay=0, gamma=0.95, epsilon=0.1):
        # parameters init
        self.n_state = n_state
        self.n_action = n_action

        self.gamma = gamma

        self.values = Network(n_state=n_state, n_action=n_action)
        self.optimizer = torch.optim.Adam(self.values.parameters(), lr=learning_rate,
                                          betas=betas, weight_decay=weighted_decay)
        self.loss_func = nn.MSELoss()

    def learn(self, state, action, reward, state_):
        state_value = self.values(torch.FloatTensor([state]))
        state_value_ = self.values(torch.FloatTensor([state_]))
        returns = reward + self.gamma * state_value_
        loss = self.loss_func(input=state_value, target=returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return returns - state_value


if __name__ == '__main__':
    env = gym.make('CartPole')
    actor = Actor(n_state=env.observation_space.shape[0], n_action=env.action_space.n, learning_rate=0.005)
    critic = Critic(n_state=env.observation_space.shape[0], n_action=env.action_space.n, learning_rate=0.005)
    env.reset()

    for episode in range(5000):
        observation = env.reset()
        state = observation
        reward = 0
        for t in range(500):
            if episode > 1000:
                env.render()
            state_before = state
            action = actor.choose_action(state=state)
            observation, _, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation
            # use the reward as Morvan Zhou defined
            # https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            state = observation
            if done:
                print("Episode {}: finished after {} timesteps".format(episode, t+1))
                break
            advantage = critic.learn(state=state_before, action=action, reward=reward, state_=state)
            actor.learn(state=state_before, action=action, advantage=advantage)

    env.close()
