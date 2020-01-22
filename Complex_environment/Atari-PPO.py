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
from PIL import Image


class AtariNet(nn.Module):
    def __init__(self, output_shape):
        super(AtariNet, self).__init__()
        self.discrete = True
        self.conv = nn.Sequential(      # input_shape: 4*84*84
            nn.Conv2d(
                in_channels=4,          # Gray: 1 channels * 4 images/
                out_channels=16,        # filters' number
                kernel_size=8,          # kernel's size
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU()
        )
        self.feature_layer = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU()
        )
        self.state_value = nn.Linear(256, 1)
        self.policy = nn.Linear(256, output_shape)

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


# def preprocess(obs):
#     for index in range(4):
#         if index == 0:
#             result = transforms.ToTensor()(obs.frame(index))
#         else:
#             temp = transforms.ToTensor()(obs.frame(index))
#             result = torch.cat((result, temp), dim=0)
#     return result       # shape: 4*84*84

def preprocess(obs):
    return torch.from_numpy(obs).float()


class Worker:
    def __init__(self, agent):
        self.agent = agent

    def cal_adv(self, states, rewards, value_pred, dones, gamma):
        """
        :param states: Tensor[T, state_features]
        :param rewards: Tensor[T]
        :param value_pred: Tensor[T+1]
        :param dones: Tensor[T+1]
        :param gamma: float
        :return: advantages -> Tensor[T] & value_target -> Tensor[T]
        """
        T = states.shape[0]
        advantages = torch.zeros(T)
        last_adv = 0
        for t in reversed(range(T)):
            non_terminal = 1 - dones[t + 1]
            delta = rewards[t] + gamma * value_pred[t + 1] * non_terminal - value_pred[t]
            last_adv = delta + gamma * non_terminal * last_adv
            advantages[t] = last_adv
        return advantages, advantages + value_pred[:T]

    def work(self, discrete):
        # env = wrappers.make_atari("BreakoutNoFrameskip-v4")
        # env = wrappers.wrap_deepmind(env, frame_stack=True)
        env = gym.make('Pendulum-v0')
        LEARN = True
        device = torch.device("cuda:0")

        episode, t, T, total_reward, horizon = 0, 0, 0, 0, 32
        done = True
        reward_array = np.array([])

        observation = env.reset()
        state = preprocess(observation)

        # state_collections = torch.zeros(horizon, 4, 84, 84)
        state_collections = torch.zeros(horizon, env.observation_space.shape[0])
        action_collections = torch.zeros(horizon)
        # action_collections = torch.zeros(horizon, env.action_space.shape[0])
        reward_collections = torch.zeros(horizon)
        done_collections = torch.zeros(horizon + 1)
        value_collections = torch.zeros(horizon + 1)
        while True:
            if done:
                observation = env.reset()
                state = preprocess(observation)
                print("Episode {}: finished after {} timesteps".format(episode, t - T))
                print("            total_reward -> {}".format(total_reward))
                T = t
                episode += 1
                total_reward = 0
            if LEARN is False:
                env.render()
            action = self.agent.choose_action(state.unsqueeze(0).to(device))
            # print('action: {}'.format(action))
            if discrete:
                observation, reward, done, info = env.step(int(action))
                if reward > 1:
                    reward = 1
                elif reward < -1:
                    reward = -1
            else:
                observation, reward, done, info = env.step([float(action)])
                if reward > 1:
                    reward = 1
                elif reward < -1:
                    reward = -1
            # img = Image.fromarray(observation.frame(0))
            # img.show()

            if t > 0 and t % horizon == 0:
                done_collections[horizon] = done
                value_collections[horizon] = self.agent.net.forward(state.unsqueeze(0).to(device))[1]
                advantages, value_target = self.cal_adv(states=state_collections, rewards=reward_collections,
                                                        value_pred=value_collections, dones=done_collections,
                                                        gamma=0.99)
                self.agent.learn(state_collections=state_collections, action_collections=action_collections,
                                 advantage_collections=advantages, value_target=value_target)
                yield advantages, value_target
                # state_collections = torch.zeros(horizon, 4, 84, 84)
                state_collections = torch.zeros(horizon, env.observation_space.shape[0])
                action_collections = torch.zeros(horizon)
                # action_collections = torch.zeros(horizon, env.action_space.shape[0])
                reward_collections = torch.zeros(horizon)
                done_collections = torch.zeros(horizon + 1)
                value_collections = torch.zeros(horizon + 1)
            state_collections[t % horizon] = state
            action_collections[t % horizon] = action
            reward_collections[t % horizon] = reward
            done_collections[t % horizon] = done
            value_collections[t % horizon] = self.agent.net.forward(state.unsqueeze(0).to(device))[1]

            state = preprocess(observation)  # next state
            t += 1
            total_reward += reward


if __name__ == '__main__':
    # env = wrappers.make_atari("BreakoutNoFrameskip-v4")
    # env = wrappers.wrap_deepmind(env, frame_stack=True)
    env = gym.make('Pendulum-v0')
    device = torch.device("cuda:0")
    LEARN = True
    n_state = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        discrete = False
        n_action = env.action_space.shape[0]
    else:
        discrete = True
        n_action = env.action_space.n
    ppo = PPO.Model(net=PPO.Net, learn=LEARN, device=device, n_state=n_state, n_action=n_action, discrete=discrete,
                    learning_rate=0.00025, epsilon=0.2)
    worker = Worker(agent=ppo)
    iter_unit = worker.work(discrete=discrete)

    while True:
        adv, v_target = iter_unit.__next__()
