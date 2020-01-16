import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical


class Net(nn.Module):
    def __init__(self, n_state, n_action):
        super(Net, self).__init__()
        self.state_value = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.policy = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, n_action),
        )

    def forward(self, state):
        action_values = self.state_value(state)
        actions = self.policy(state)
        action_prob = torch.nn.functional.softmax(actions, dim=-1)
        # print('action_prob: ', action_prob)
        distribution = Categorical(action_prob)
        return distribution, action_values


class Model:
    def __init__(self, net, device, n_action, learn=False, gamma=0.95, learning_rate=0.005,
                 epsilon=0.1):
        # init networks
        if learn is True:
            self.net = net(output_shape=n_action).to(device)
        else:
            self.load_net()
        self.old_net = net(output_shape=n_action).to(device)
        self.n_action = n_action
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.device = device

    def learn(self, state_collections, action_collections, reward_collections, final_state):
        """
        :param state_collections: Tensor[T, state_features]
        :param action_collections: Tensor[T]
        :param reward_collections: Tensor[T]
        :param final_state: Tensor[state_features]
        :return:
        """
        state_collections = state_collections.to(self.device)
        action_collections = action_collections.to(self.device)
        reward_collections = reward_collections.to(self.device)
        final_state = final_state.to(self.device)

        T = state_collections.shape[0]
        returns = self.net.forward(final_state)[1]  # V(S_final)
        advantage_collections = torch.FloatTensor([])  # Tensor[T]
        return_collections = torch.FloatTensor([])  # Tensor[T]
        predict_value = self.net.forward(state_collections)[1]

        for i in range(1, T + 1):
            # t: T -> 1
            # Returns[t] = r[t] + gamma * Returns[t+1]
            returns = reward_collections[T - i] + self.gamma * returns
            advantage = returns - self.net.forward(state_collections[T - i].unsqueeze(0))[1]
            if advantage_collections.shape[0] == 0:
                advantage_collections = advantage.unsqueeze(0)
            else:
                advantage_collections = torch.cat((advantage_collections, advantage.unsqueeze(0)))
            if return_collections.shape[0] == 0:
                return_collections = returns.unsqueeze(0)
            else:
                return_collections = torch.cat((return_collections, returns.unsqueeze(0)))

        idx = [i for i in range(advantage_collections.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(device=self.device)
        advantage_collections = advantage_collections.index_select(0, idx)

        self.old_net.load_state_dict(self.net.state_dict())
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.learning_rate)
        for j in range(12):
            # loss = mean(ratio * advantages) - lambda * KL(old_net, net)
            # J = -loss
            # print(torch.exp(self.policy.forward(state_collections).log_prob(action_collections)))
            ratio = torch.div(torch.exp(self.net.forward(state_collections)[0].log_prob(action_collections)),
                              torch.exp(self.old_net.forward(state_collections)[0].log_prob(action_collections)))
            policy_loss = - torch.mean(torch.min(torch.mul(ratio, advantage_collections.detach()),
                                                 torch.mul(torch.clamp(ratio, min=1-self.epsilon, max=1+self.epsilon),
                                                           advantage_collections.detach())
                                                 )
                                       )
            optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            optimizer.step()

            critic_loss = torch.mean(torch.pow(predict_value - return_collections, 2))

            optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            optimizer.step()

    def choose_action(self, state):
        # print(self.policy.forward(state).probs)
        action = self.net.forward(state)[0].sample()
        return action

    def save_net(self):
        torch.save(self.net, 'params/ppo_net.pkl')

    def load_net(self):
        self.net = torch.load('params/ppo_net.pkl')


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    LEARN = False
    model = Model(net=Net, learn=LEARN, n_action=env.action_space.n, learning_rate=0.0005, epsilon=0.1)
    env.reset()
    BATCH_SIZE = 32
    episode = 0
    T = 0
    done = True
    state_collections = torch.FloatTensor([[]])
    action_collections = torch.FloatTensor([])
    reward_collections = torch.FloatTensor([])

    for t in range(500000):
        if done:
            observation = env.reset()
            state = observation
            reward = 0
        if LEARN is False:
            env.render()
        action = model.choose_action(torch.FloatTensor(state))
        # print('action: {}'.format(action))
        observation, reward, done, info = env.step(int(action))
        if done:
            reward = 0
        else:
            reward = 1
        # x, x_dot, theta, theta_dot = observation
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

        state = observation  # next state
        if (t - T) % BATCH_SIZE == 0:
            if LEARN is True:
                model.learn(state_collections=state_collections, action_collections=action_collections,
                            reward_collections=reward_collections, final_state=torch.FloatTensor(state))
            state_collections = torch.FloatTensor([[]])
            action_collections = torch.FloatTensor([])
            reward_collections = torch.FloatTensor([])

        if done:
            if LEARN is True and reward_collections.shape[0] != 0:
                model.learn(state_collections=state_collections, action_collections=action_collections,
                            reward_collections=reward_collections, final_state=torch.FloatTensor(state))
            state_collections = torch.FloatTensor([[]])
            action_collections = torch.FloatTensor([])
            reward_collections = torch.FloatTensor([])
            print("Episode {}: finished after {} timesteps".format(episode, t - T))
            if (t - T) > 450:
                model.save_net()
            T = t
            episode += 1

    env.close()