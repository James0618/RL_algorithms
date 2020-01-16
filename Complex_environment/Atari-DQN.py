import sources.DQN as DQN
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gym
import time
import cv2
import numpy as np
import common.atari_wrappers as wrappers


class Network(nn.Module):
    def __init__(self, output_shape):
        super(Network, self).__init__()
        self.conv = nn.Sequential(      # input_shape: 2*84*84
            nn.Conv2d(
                in_channels=4,          # Gray: 1 channels * 4 images/
                out_channels=16,        # filters' number
                kernel_size=8,          # kernel's size
                stride=4,
            ),                          # output_shape: 16*20*20
            nn.ReLU(),
            nn.Conv2d(                  # input_shape: 16*20*20
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,               # output_shape: 32*9*9
            ),
            nn.ReLU(),
        )
        self.fcn = nn.Sequential(
            nn.Linear(32*9*9, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape)
        )

    def forward(self, state):
        output = self.conv(state)
        output = output.view(output.size(0), -1)
        output = self.fcn(output)
        return output


def preprocess(obs):
    for index in range(4):
        if index == 0:
            result = transforms.ToTensor()(obs.frame(index)).unsqueeze(0)
        else:
            temp = transforms.ToTensor()(obs.frame(index)).unsqueeze(0)
            result = torch.cat((result, temp), dim=1)
    return result


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    # env = wrappers.make_atari("Breakout-v0")
    env = wrappers.wrap_deepmind(env, frame_stack=True)
    LEARN = True
    net = Network(env.action_space.n)
    device = torch.device("cuda:0")
    net = net.to(device)
    agent = DQN.DQN(network=net, device=device, n_replay=10000, n_action=env.action_space.n, learning_rate=0.00025,
                    epsilon=0.1, gamma=0.99, learn=LEARN)
    env.reset()
    reward_array = np.array([])
    for episode in range(250000):
        observation = env.reset()
        state = preprocess(observation)
        total_reward = 0
        total_q_value = 0
        t = 0
        while True:
            if LEARN is False:
                env.render()
            state_before = state
            action, q_value = agent.choose_action(state)
            total_q_value += q_value
            # print('action: {}'.format(action))
            observation, reward, done, info = env.step(int(action))
            t += 1
            if LEARN is False:
                time.sleep(0.05)
            state = preprocess(observation)           # generate next state
            # if t == 600 - 1:
            #     done = True

            transition = [state_before, action, reward, state, done]
            total_reward += reward
            agent.store_transition(transition=transition)
            if done:
                print("Episode {}: Reward -> {} after {} steps".format(episode, total_reward, t+1))
                print("            Average Q values: {}".format(total_q_value/t))
                reward_array = np.append(reward_array, np.array([total_reward]))
                np.save('params/result-atari.npy', reward_array)
                t = 0
                break

            # learn when replay has enough transitions
            if agent.full is True:
                if t % 4 == 0 and LEARN is True:
                    agent.learn()

            # save success params
            if total_reward > 10:
                agent.save_net("dqn-atari")
