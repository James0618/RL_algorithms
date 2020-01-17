import sources.PPO as PPO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.distributions import Categorical
import gym
import time
import cv2
import numpy as np
import common.atari_wrappers as wrappers


class Network(nn.Module):
    def __init__(self, output_shape):
        super(Network, self).__init__()
        self.conv = nn.Sequential(      # input_shape: 4*84*84
            nn.Conv2d(
                in_channels=4,          # Gray: 1 channels * 4 images/
                out_channels=32,        # filters' number
                kernel_size=8,          # kernel's size
                stride=4,
                padding=2
            ),                          # output_shape: 32*
            nn.ReLU(),
            nn.Conv2d(                  # input_shape: 32*
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,               # output_shape: 64*
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(                  # input_shape: 64*
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,               # output_shape: 64*10*10
                padding=1
            ),
            nn.ReLU()
        )
        self.feature_layer = nn.Sequential(
            nn.Linear(6400, 512),
            nn.ReLU()
        )
        self.state_value = nn.Linear(512, 1)
        self.policy = nn.Linear(512, output_shape)

    def forward(self, state):
        output = self.conv(state)
        features = output.view(output.size(0), -1)
        features = self.feature_layer(features)
        action_values = self.state_value(features)
        actions = self.policy(features)
        action_prob = torch.nn.functional.softmax(actions, dim=-1)
        # print('action_prob: ', action_prob)
        distribution = Categorical(action_prob)
        return distribution, action_values


def preprocess(obs):
    for index in range(4):
        if index == 0:
            result = transforms.ToTensor()(obs.frame(index)).unsqueeze(0)
        else:
            temp = transforms.ToTensor()(obs.frame(index)).unsqueeze(0)
            result = torch.cat((result, temp), dim=1)
    return result


if __name__ == '__main__':
    # env = gym.make("BreakoutNoFrameskip-v4")
    env = wrappers.make_atari("BreakoutNoFrameskip-v4")
    env = wrappers.wrap_deepmind(env, frame_stack=True)
    LEARN = True
    device = torch.device("cuda:0")
    model = PPO.Model(net=Network, device=device, learn=LEARN, n_action=env.action_space.n, learning_rate=0.0001,
                      gamma=0.999, epsilon=0.2)
    env.reset()
    BATCH_SIZE = 64
    episode = 0
    T = 0
    done = True
    state_collections = torch.FloatTensor([[]])
    action_collections = torch.FloatTensor([])
    reward_collections = torch.FloatTensor([])
    total_reward = 0
    reward_array = np.array([])

    for t in range(5000000):
        if done:
            observation = env.reset()
            state = preprocess(observation)
            reward = 0
            total_reward = 0
        if LEARN is False:
            env.render()
        action = model.choose_action(torch.FloatTensor(state).to(device))
        # print('action: {}'.format(action))
        observation, reward, done, info = env.step(int(action))
        total_reward += reward

        if state_collections.shape[1] == 0:
            state_collections = torch.FloatTensor(state)
        else:
            state_collections = torch.cat((state_collections, torch.FloatTensor(state)))
        if action_collections.shape[0] == 0:
            action_collections = action.unsqueeze(0)
        else:
            action_collections = torch.cat((action_collections, action.unsqueeze(0)))
        if reward_collections.shape[0] == 0:
            reward_collections = torch.FloatTensor([reward])
        else:
            reward_collections = torch.cat((reward_collections, torch.FloatTensor([reward])))

        state = preprocess(observation)  # next state

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
            print("Episode {}: Reward -> {} after {} steps".format(episode, total_reward, t - T))
            T = t
            total_reward = 0
            reward_array = np.append(reward_array, np.array([total_reward]))
            np.save('params/result-atari-ppo.npy', reward_array)
            episode += 1

    env.close()
