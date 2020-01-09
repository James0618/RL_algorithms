import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym


class CriticNet(nn.Module):
    def __init__(self, n_state):
        super(CriticNet, self).__init__()
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


class ActorNet(nn.Module):
    def __init__(self, n_state, n_action):
        super(ActorNet, self).__init__()
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
        return action_prob


class Actor:
    def __init__(self, n_state, n_action, learning_rate=0.001):
        # parameters init
        self.n_state = n_state
        self.n_action = n_action

        self.policy = ActorNet(n_state=n_state, n_action=n_action)
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=learning_rate)

    def choose_action(self, state):
        policy = self.policy(torch.FloatTensor([state]))
        # print('policy: {}'.format(torch.softmax(policy, dim=-1)))
        actions_distribution = Categorical(policy)
        action = actions_distribution.sample()
        return action

    def __loss_func(self, state, advantage, action):
        policy = self.policy(torch.FloatTensor([state]))
        actions_distribution = Categorical(policy)
        log_probability = actions_distribution.log_prob(action)
        actor_loss = - log_probability * advantage
        # print('advantage: {}, reward: {}, log_probability: {}'.format(advantage, reward, log_probability))
        return actor_loss

    def learn(self, state, advantage, action):
        loss = self.__loss_func(state=state, advantage=advantage.detach(), action=action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Critic:
    def __init__(self, n_state, learning_rate=0.001, gamma=0.95):
        # parameters init
        self.n_state = n_state
        self.gamma = gamma
        self.values = CriticNet(n_state=n_state)
        self.optimizer = torch.optim.SGD(self.values.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def learn(self, state, reward, state_):
        state_value = self.values(torch.FloatTensor(state))
        state_value_ = self.values(torch.FloatTensor(state_))
        returns = reward + self.gamma * state_value_
        loss = self.loss_func(input=returns, target=state_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print('return: {}, state_value: {}'.format(returns, state_value))
        return returns - state_value


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    actor = Actor(n_state=2*env.observation_space.shape[0], n_action=env.action_space.n, learning_rate=0.005)
    critic = Critic(n_state=2*env.observation_space.shape[0], learning_rate=0.005)
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
                action = actor.choose_action(state=state)
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
                advantage = critic.learn(state=state_before, reward=reward, state_=state)
                actor.learn(state=state_before, action=action, advantage=advantage)
        print("Episode {}: finished after {} timesteps".format(episode, 'max'))

    env.close()
