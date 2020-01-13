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
                in_channels=4,          # Gray: 1 channels * 4 images
                groups=4,               # 1 state -> 4 images
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


def preprocess(ob):
    ob = cv2.cvtColor(cv2.resize(ob, (84, 110)), cv2.COLOR_BGR2GRAY)
    ob = ob[26:110, :]
    # ret, ob = cv2.threshold(ob, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(ob, (84, 84, 1))


if __name__ == '__main__':
    # env = gym.make("SpaceInvaders-v0")
    env = wrappers.make_atari("SpaceInvaders-v0", max_episode_steps=600, max_and_skip=True)
    env = wrappers.wrap_deepmind(env)
    LEARN = False
    net = Network(env.action_space.n)
    device = torch.device("cuda:0")
    net = net.to(device)
    agent = DQN.DQN(network=net, device=device, n_replay=16000, n_action=env.action_space.n, learning_rate=0.001,
                    epsilon=0.1, learn=LEARN)
    observation = env.reset()
    observation = transforms.ToTensor()(observation).unsqueeze(0)
    env.reset()
    reward_array = np.array([])
    for episode in range(100000):
        observation = env.reset()
        # observation = preprocess(observation)
        observation = transforms.ToTensor()(observation).unsqueeze(0)
        state = observation
        total_reward = 0
        for t in range(1000):
            if LEARN is False:
                env.render()
            if t < 3:
                observation, reward, done, info = env.step(env.action_space.sample())
                # observation = preprocess(observation)
                observation = transforms.ToTensor()(observation).unsqueeze(0)   # 1*1*84*84
                state = torch.cat((state, observation), dim=1)                  # 1*4*84*84
                total_reward += reward
            else:
                state_before = state
                action = agent.choose_action(state)
                # print('action: {}'.format(action))
                observation, reward, done, info = env.step(int(action))
                if LEARN is False:
                    time.sleep(0.01)
                # observation = preprocess(observation)
                observation = transforms.ToTensor()(observation).unsqueeze(0)
                indices = torch.LongTensor([1, 2, 3])
                state = torch.cat((torch.index_select(state, 1, indices), observation), dim=1)  # generate next state
                if t == 600 - 1:
                    done = True

                transition = [state_before, action, reward, state]
                total_reward += reward
                agent.store_transition(transition=transition)
                if done:
                    print("Episode {}: Reward -> {} after {} steps".format(episode, total_reward, t+1))
                    print("        left lives: {}".format(info["ale.lives"]))
                    reward_array = np.append(reward_array, np.array([total_reward]))
                    np.save('params/result-atari.npy', reward_array)
                    break

                # learn when replay has enough transitions
                if episode >= 10:
                    if t % 20 == 0 and LEARN is True:
                        agent.learn()

                # save success params
                if total_reward > 15:
                    agent.save_net("dqn-atari")
