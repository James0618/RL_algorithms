import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical


class ValueNet(nn.Module):
    def __init__(self, n_state):
        super(ValueNet, self).__init__()
        self.state_value = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        action_prob = torch.nn.functional.softmax(actions, dim=0)
        # print('action_prob: ', action_prob)
        distribution = Categorical(action_prob)
        return distribution


class Model:
    def __init__(self, n_state, n_action, gamma=0.95, learning_rate=0.005, epsilon=0.2):
        # init networks
        self.policy = PolicyNet(n_state=n_state, n_action=n_action)
        self.old_policy = PolicyNet(n_state=n_state, n_action=n_action)
        self.state_value = ValueNet(n_state=n_state)

        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def learn(self, state_collections, action_collections, reward_collections, final_state):
        """
        :param state_collections: Tensor[T, state_features]
        :param action_collections: Tensor[T]
        :param reward_collections: Tensor[T]
        :param final_state: Tensor[state_features]
        :return:
        """
        T = state_collections.shape[0]
        returns = self.state_value.forward(final_state)  # V(S_final)
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
        state_value_optimizer = torch.optim.SGD(params=self.state_value.parameters(), lr=self.learning_rate)
        # loss_func = nn.KLDivLoss()
        for j in range(8):
            # loss = mean(ratio * advantages) - lambda * KL(old_net, net)
            # J = -loss
            ratio = torch.div(self.policy.forward(state_collections).log_prob(action_collections),
                              old_policy.forward(state_collections).log_prob(action_collections))
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

        for j in range(8):
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

        # print(old_policy(state_collections).probs)
        # if loss_func(torch.log(old_policy.forward(state_collections).probs),
        #              torch.log(self.policy.forward(state_collections).probs)) > self.kl_target * 1.5:
        #     self.lamb = self.lamb * 2
        # elif loss_func(torch.log(old_policy.forward(state_collections).probs),
        #                torch.log(self.policy.forward
        #                              (state_collections).probs)) < self.kl_target / 1.5:
        #     self.lamb = self.lamb / 2

    def choose_action(self, state):
        # print(self.policy.forward(state).probs)
        action = self.policy.forward(state).sample()
        return action


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    model = Model(n_state=2*env.observation_space.shape[0], n_action=env.action_space.n, learning_rate=0.0001)
    env.reset()
    BATCH_SIZE = 16

    for episode in range(50000):
        observation = env.reset()
        state = observation

        reward = 0
        state_collections = torch.FloatTensor([[]])
        action_collections = torch.FloatTensor([])
        reward_collections = torch.FloatTensor([])
        for t in range(1000):
            if episode > 40000:
                env.render()
            if t < 1:
                observation, _, done, info = env.step(0)
                state = np.append(state, observation)
            else:
                action = model.choose_action(torch.FloatTensor(state))
                # print('action: {}'.format(action))
                observation, _, done, info = env.step(int(action))

                x, x_dot, theta, theta_dot = observation
                # use the reward as Morvan Zhou defined
                # https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                reward = r1 + r2

                # store [state, action, reward] collections
                if state_collections.shape[1] == 0:
                    state_collections = torch.FloatTensor(state).unsqueeze(0)
                else:
                    state_collections = torch.cat((state_collections, torch.FloatTensor(state).unsqueeze(0)))
                if action_collections.shape[0] == 0:
                    action_collections = action.unsqueeze(0)
                else:
                    action_collections = torch.cat((action_collections, action.unsqueeze(0)))

                if reward_collections.shape[0] == 0:
                    reward_collections = torch.FloatTensor([reward])
                else:
                    reward_collections = torch.cat((reward_collections, torch.FloatTensor([reward])))

                state = np.append(state[-4:], observation)  # next state
                if t == 1000 - 1:
                    done = True
                if t % BATCH_SIZE == 0:
                    model.learn(state_collections=state_collections, action_collections=action_collections,
                                reward_collections=reward_collections, final_state=torch.FloatTensor(state))
                    state_collections = torch.FloatTensor([[]])
                    action_collections = torch.FloatTensor([])
                    reward_collections = torch.FloatTensor([])
                    if done:
                        break

                if done:
                    model.learn(state_collections=state_collections, action_collections=action_collections,
                                reward_collections=reward_collections, final_state=torch.FloatTensor(state))
                    print("Episode {}: finished after {} timesteps".format(episode, t + 1))
                    state_collections = torch.FloatTensor([[]])
                    action_collections = torch.FloatTensor([])
                    reward_collections = torch.FloatTensor([])
                    break

    env.close()
