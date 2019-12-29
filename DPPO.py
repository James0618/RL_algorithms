import torch
import torch.nn as nn
import numpy as np
import gym
from PPO import Model
import torch.multiprocessing as mp


class Worker(mp.Process):
    def __init__(self, global_ppo, env, name, learn):
        super(Worker, self).__init__()
        self.name = name
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
                if episode > 1500 or LEARN is False:
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
                        if LEARN is True:
                            self.ppo.learn(state_collections=state_collections, action_collections=action_collections,
                                           reward_collections=reward_collections, final_state=torch.FloatTensor(state))
                        state_collections = torch.FloatTensor([[]])
                        action_collections = torch.FloatTensor([])
                        reward_collections = torch.FloatTensor([])
                        if done:
                            break

                    if done:
                        if LEARN is True:
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
    global_net = Model(n_state=2*env.observation_space.shape[0], learn=LEARN, n_action=env.action_space.n,
                       learning_rate=0.005, epsilon=0.1)
