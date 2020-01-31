import sources.PPO as PPO
import torch
import torch.nn as nn
import torch.multiprocessing as mp
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
        self.conv = nn.Sequential(  # input_shape: 4*84*84
            nn.Conv2d(
                in_channels=4,  # Gray: 1 channels * 4 images/
                out_channels=16,  # filters' number
                kernel_size=8,  # kernel's size
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
            nn.Linear(2592, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
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
    def __init__(self, agent, worker_id):
        super(Worker, self).__init__()
        self.agent = agent
        self.worker_id = worker_id
        self.horizon = 32

    def cal_adv(self, states, rewards, dones, final_state, gamma):
        """
        :param states: Tensor[T, state_features]
        :param rewards: Tensor[T]
        :param dones: Tensor[T+1]
        :param gamma: float
        :return: advantages -> Tensor[T] & value_target -> Tensor[T]
        """
        T = states.shape[0]
        advantages = torch.zeros(T)
        last_adv = 0
        value_pred = torch.zeros(self.horizon + 1)
        # calculate value_predict
        for i in range(self.horizon):
            value_pred[i] = self.agent.net.forward(states[i].unsqueeze(0).to(device))[1]
        value_pred[self.horizon] = self.agent.net.forward(final_state.unsqueeze(0).to(device))[1]

        for t in reversed(range(T)):
            non_terminal = 1 - dones[t + 1]
            delta = rewards[t] + gamma * value_pred[t + 1] * non_terminal - value_pred[t]
            last_adv = delta + gamma * non_terminal * last_adv      # * 0.95
            advantages[t] = last_adv
        value_target = advantages + value_pred[:T]
        # advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        return advantages, value_target

    def work(self, discrete, barrier, queue):
        # env = wrappers.make_atari("BreakoutNoFrameskip-v4")
        # env = wrappers.wrap_deepmind(env, frame_stack=True)
        env = gym.make('CartPole-v1')
        LEARN = True
        device = torch.device("cpu")
        global episode
        t, T, total_reward, horizon = 0, 0, 0, self.horizon
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
            else:
                observation, reward, done, info = env.step([float(action)])
                # reward = (reward + 8) / 8
            # img = Image.fromarray(255 * state.numpy()[0])
            # img.show()

            if t > 0 and t % horizon == 0:
                done_collections[horizon] = done
                # yield state_collections, action_collections, done_collections, reward_collections, state
                queue.put([state_collections, action_collections, done_collections, reward_collections, state])
                barrier.wait()
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
    env = gym.make('CartPole-v1')
    device = torch.device("cpu")
    LEARN = True
    i, episode = 0, 0
    n_state = env.observation_space.shape[0]
    barrier = mp.Barrier(2)
    queue = mp.Queue()
    if isinstance(env.action_space, gym.spaces.Box):
        discrete = False
        n_action = env.action_space.shape[0]
    else:
        discrete = True
        n_action = env.action_space.n

    net = PPO.Net(n_state, n_action, discrete).to(device)
    net.share_memory()
    old_net = PPO.Net(n_state, n_action, discrete).to(device)
    old_net.share_memory()
    ppo = PPO.Model(net=net, old_net=old_net, learn=LEARN, device=device, n_state=n_state, n_action=n_action, discrete=discrete,
                    learning_rate=0.00005, epsilon=0.1)
    workers = [Worker(agent=ppo, worker_id=i) for i in range(1)]
    process_list = []
    for worker in workers:
        process = mp.Process(target=worker.work, args=(discrete, barrier, queue))
        process_list.append(process)
        process.start()

    while i < 100000:
        for worker in workers:
            temp = queue.get()
            s, a, d_collect, r_collect, final_s = temp[0], temp[1], temp[2], temp[3], temp[4]
            ad, vt = worker.cal_adv(states=s, rewards=r_collect, dones=d_collect, final_state=final_s, gamma=0.99)
            agent_id, loss = worker.agent.learn(state_collections=s, action_collections=a, advantage_collections=ad,
                                                value_target=vt)
            barrier.wait()
        i += 1

    for process in process_list:
        process.join()
