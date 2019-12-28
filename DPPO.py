from PPO import Model
import torch
import torch.nn as nn
import gym


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
