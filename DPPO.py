import torch
import torch.multiprocessing as mp
import numpy as np
import gym
from torch.distributions import Categorical
from PPO import PolicyNet
from DPPO_learner import Agent


class Worker(mp.Process):
    def __init__(self, policy, env, worker_id, state_queue, action_queue, reward_queue, final_queue, global_episode,
                 lock, learn=True):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.policy = policy
        self.env = env
        self.BATCH_SIZE = 16
        self.LEARN = learn
        self.MAX_EPISODE = 3000

        self.state_queue = state_queue
        self.action_queue = action_queue
        self.reward_queue = reward_queue
        self.final_queue = final_queue
        self.lock = lock
        self.global_episode = global_episode

    def run(self):
        while self.global_episode.value < self.MAX_EPISODE:
            observation = env.reset()
            state = observation

            reward = 0
            state_collections = torch.FloatTensor([[]])
            action_collections = torch.FloatTensor([])
            reward_collections = torch.FloatTensor([])
            for t in range(500):
                if self.LEARN is False:
                    env.render()
                if t < 1:
                    observation, _, done, info = env.step(0)
                    state = np.append(state, observation)
                else:
                    action = self.policy.forward(state).sample()
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
                            self.state_queue.put(state_collections)
                            self.action_queue.put(action_collections)
                            self.reward_queue.put(reward_collections)
                            self.final_queue.put(torch.FloatTensor(state))
                        state_collections = torch.FloatTensor([[]])
                        action_collections = torch.FloatTensor([])
                        reward_collections = torch.FloatTensor([])
                        if done:
                            break

                    if done:
                        if self.LEARN is True:
                            self.state_queue.put(state_collections)
                            self.action_queue.put(action_collections)
                            self.reward_queue.put(reward_collections)
                            self.final_queue.put(torch.FloatTensor(state))
                        print("Episode {}: finished after {} timesteps".format(self.global_episode.value, t + 1))
                        self.lock.acquire()
                        self.global_episode.value += 1
                        self.lock.release()
                        state_collections = torch.FloatTensor([[]])
                        action_collections = torch.FloatTensor([])
                        reward_collections = torch.FloatTensor([])
                        break


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    LEARN = True
    NUMBER = int(mp.cpu_count() / 2)
    global_episode = mp.Value('i', 0)
    MAX_EPISODE = 3000
    state_queue = mp.Queue()
    action_queue = mp.Queue()
    reward_queue = mp.Queue()
    final_queue = mp.Queue()
    lock = mp.Lock()

    policy = PolicyNet(n_state=2*env.observation_space.shape[0], n_action=env.action_space.n)
    policy.share_memory()

    agent = Agent(policy=policy, n_state=env.observation_space.shape[0], n_action=env.action_space.n,
                  state_queue=state_queue, action_queue=action_queue, reward_queue=reward_queue,
                  final_queue=final_queue, global_episode=global_episode)
    # TODO: workers & agent, start & join
    workers = [Worker(policy=policy, env=env, worker_id=i, state_queue=state_queue, action_queue=action_queue,
                      reward_queue=reward_queue, final_queue=final_queue, global_episode=global_episode,
                      lock=lock, learn=True) for i in range(NUMBER)]
    agent.start()
    for worker in workers:
        worker.start()

    while True:
        if global_episode.value >= MAX_EPISODE:
            break

    for worker in workers:
        worker.join()
    agent.join()
