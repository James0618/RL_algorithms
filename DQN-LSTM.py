import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import gym


class Network(nn.Module):
    def __init__(self, n_state, n_action):
        super(Network, self).__init__()
        self.num_layers = 1
        self.n_state = n_state
        self.n_action = n_action
        # model init
        self.lstm_layer = nn.LSTM(input_size=n_state, hidden_size=n_state, num_layers=self.num_layers)
        self.hidden2action = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_action),
        )

    def forward(self, state):
        batch_size = state.shape[0]
        state_lstm = state.unsqueeze(0)
        # h0 = torch.FloatTensor(torch.randn(self.num_layers, batch_size, self.n_state))
        # c0 = torch.FloatTensor(torch.randn(self.num_layers, batch_size, self.n_state))
        output, _ = self.lstm_layer(state_lstm)
        out = output.squeeze()
        action_values = self.hidden2action(out)
        return action_values


class DQN:
    def __init__(self, n_replay, n_state, n_action, learn=True, learning_rate=0.005, gamma=0.95, epsilon=0.1):
        # parameters init
        self.n_replay = n_replay
        self.n_state = n_state
        self.n_action = n_action

        if learn is False:
            self.load_net()
        else:
            self.q_net = Network(n_state=self.n_state, n_action=self.n_action)

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
            # action = self.q_net.forward(state_tensor).max(0)[1]
            action = torch.argmax(self.q_net.forward(state_tensor))
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
        data_array = np.array(transitions, dtype=float)
        data_state = torch.from_numpy(data_array[:, 0:self.n_state]).float()
        data_action = torch.from_numpy(data_array[:, self.n_state:self.n_state+1]).long()
        data_reward = torch.from_numpy(data_array[:, self.n_state+1:self.n_state+2]).float()
        data_state_ = torch.from_numpy(data_array[:, -self.n_state:]).float()

        # calculate q_current and q_next
        q_value = self.q_net(data_state).gather(1, data_action)
        q_next = self.q_net(data_state_).max(1)[0].view(len(transitions), 1)
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


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    LEARN = True
    agent = DQN(n_replay=10000, n_action=2, n_state=4, learning_rate=0.01, learn=LEARN)

    env.reset()
    for episode in range(50000):
        observation = env.reset()
        state = observation
        reward = 0
        for t in range(500):
            # if episode > 3000 or LEARN is False:
            #     env.render()
            state_before = state
            action = agent.choose_action(state)
            observation, _, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation
            # use the reward as Morvan Zhou defined
            # https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # reward = r1 + r2
            if done:
                reward = 0
            else:
                reward = 1

            state = observation
            if done:
                print("Episode {}: finished after {} timesteps".format(episode, t+1))
                break
            agent.store_transition(state.tolist() + [action] + [reward] + state_before.tolist())

            # learn when replay has enough transitions
            if episode >= 5:
                if t % 2 == 0 and LEARN is True:
                    agent.learn()

            # save success params
            if t > 400:
                if episode % 20 == 0:
                    agent.save_net()

    env.close()
