import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import gym
from torch.distributions import Categorical


class DPPO(nn.Module):
    def __init__(self, n_state, n_action):
        super(DPPO, self).__init__()
        self.state_value = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.policy = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_action),
        )

    def forward(self, state):
        action_values = self.state_value(state)
        actions = self.policy(state)
        action_prob = torch.nn.functional.softmax(actions, dim=-1)
        # print('action_prob: ', action_prob)
        distribution = Categorical(action_prob)
        return action_values, distribution

    def choose_action(self, state):
        self.eval()
        _, distribution = self.forward(state)
        return distribution.sample()


class Worker(mp.Process):
    def __init__(self, global_ppo, env, worker_id, learn):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.ppo = global_ppo
        self.env = env
        self.BATCH_SIZE = 16
        self.LEARN = learn

    def run(self):
        for episode in range(2000):
            observation = env.reset()
            state = observation

            reward = 0
            state_collections = torch.FloatTensor([[]])
            action_collections = torch.FloatTensor([])
            reward_collections = torch.FloatTensor([])
            for t in range(500):
                if episode > 1500 or self.LEARN is False:
                    env.render()
                if t < 1:
                    observation, _, done, info = env.step(0)
                    state = np.append(state, observation)
                else:
                    action = self.ppo.choose_action(torch.FloatTensor(state))
                    # print('action: {}'.format(action))
                    observation, _, done, info = env.step(int(action))

                    x, x_dot, theta, theta_dot = observation
                    if done:
                        reward = 0
                    else:
                        reward = 1

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
                    if t == 500 - 1:
                        done = True
                    if t % self.BATCH_SIZE == 0:
                        if self.LEARN is True:
                            self.ppo.learn(state_collections=state_collections, action_collections=action_collections,
                                           reward_collections=reward_collections, final_state=torch.FloatTensor(state))
                        state_collections = torch.FloatTensor([[]])
                        action_collections = torch.FloatTensor([])
                        reward_collections = torch.FloatTensor([])
                        if done:
                            break

                    if done:
                        if self.LEARN is True:
                            self.ppo.learn(state_collections=state_collections, action_collections=action_collections,
                                           reward_collections=reward_collections, final_state=torch.FloatTensor(state))
                        print("Episode {}: finished after {} timesteps".format(episode, t + 1))
                        if t > 450:
                            self.ppo.save_net()
                        state_collections = torch.FloatTensor([[]])
                        action_collections = torch.FloatTensor([])
                        reward_collections = torch.FloatTensor([])
                        break


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    LEARN = True
    worker_number = int(mp.cpu_count() / 2)
    global_episode = mp.Value('i', 0)
    global_collections_queue = mp.Queue()
