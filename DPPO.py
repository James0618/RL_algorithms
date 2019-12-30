import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
import gym
import time
from torch.distributions import Categorical
from PPO import PolicyNet, ValueNet
from DPPO_learner import Agent


class Worker(mp.Process):
    def __init__(self, policy, worker_id, queue, global_episode, lock, learn=True):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.policy = policy
        self.env = env
        self.BATCH_SIZE = 16
        self.LEARN = learn
        self.MAX_EPISODE = 30000

        self.queue = queue
        self.lock = lock
        self.global_episode = global_episode

    def run(self):
        env = gym.make('CartPole-v1')
        env.reset()
        while self.global_episode.value < self.MAX_EPISODE:
            observation = env.reset()
            state = observation
            collections = []
            reward = 0
            state_collections = torch.FloatTensor([[]])
            action_collections = torch.FloatTensor([])
            reward_collections = torch.FloatTensor([])
            for t in range(500):
                # if self.LEARN is False:
                #     env.render()
                if t < 1:
                    observation, _, done, info = env.step(0)
                    state = np.append(state, observation)
                else:
                    action = self.policy.forward(torch.FloatTensor(state)).sample()
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
                            collections.append(state_collections)
                            collections.append(action_collections)
                            collections.append(reward_collections)
                            collections.append(torch.FloatTensor(state))
                            self.queue.put(collections)
                        state_collections = torch.FloatTensor([[]])
                        action_collections = torch.FloatTensor([])
                        reward_collections = torch.FloatTensor([])
                        # time.sleep(0.2)
                        if done:
                            break

                    if done:
                        if self.LEARN is True:
                            collections.append(state_collections)
                            collections.append(action_collections)
                            collections.append(reward_collections)
                            collections.append(torch.FloatTensor(state))
                            self.queue.put(collections)
                        print("Episode {}: finished after {} timesteps".format(self.global_episode.value, t + 1))
                        self.lock.acquire()
                        self.global_episode.value += 1
                        self.lock.release()
                        state_collections = torch.FloatTensor([[]])
                        action_collections = torch.FloatTensor([])
                        reward_collections = torch.FloatTensor([])
                        time.sleep(0.5)
                        break


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    LEARN = True
    NUMBER = int(mp.cpu_count() / 2)
    global_episode = mp.Value('i', 0)
    MAX_EPISODE = 30000
    queue = mp.Queue()
    lock = mp.Lock()

    policy = PolicyNet(n_state=2*env.observation_space.shape[0], n_action=env.action_space.n)
    policy.share_memory()

    agent = Agent(policy=policy, n_state=2*env.observation_space.shape[0], n_action=env.action_space.n, learn=LEARN,
                  queue=queue, global_episode=global_episode)
    workers = [Worker(policy=policy, worker_id=i, queue=queue, global_episode=global_episode,
                      lock=lock, learn=LEARN) for i in range(NUMBER)]
    agent.start()
    for worker in workers:
        worker.start()

    while True:
        if global_episode.value >= MAX_EPISODE:
            break

    for worker in workers:
        worker.join()
    agent.join()
