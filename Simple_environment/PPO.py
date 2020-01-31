import sources.PPO as PPO
import torch
import torch.multiprocessing as mp
import gym
import time
import numpy as np


def preprocess(obs):
    return torch.from_numpy(obs).float()


class Worker:
    def __init__(self, agent, worker_id):
        super(Worker, self).__init__()
        self.agent = agent
        self.worker_id = worker_id

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
        value_target = advantages + value_pred[:T]
        # advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        return advantages, value_target

    def work(self, discrete):
        env = gym.make('Pendulum-v0')
        LEARN = True
        device = torch.device("cuda:0")
        global episode
        t, T, total_reward, horizon = 0, 0, 0, 256
        done = True
        reward_array = np.array([])

        observation = env.reset()
        state = preprocess(observation)

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
            action = self.agent.choose_action(state.to(device))
            if discrete:
                observation, reward, done, info = env.step(int(action))
            else:
                observation, reward, done, info = env.step([float(action)])
                # reward = (reward + 8) / 8

            if t > 0 and t % horizon == 0:
                done_collections[horizon] = done
                value_collections[horizon] = self.agent.net.forward(state.to(device))[1]
                advantages, value_target = self.cal_adv(states=state_collections, rewards=reward_collections,
                                                        value_pred=value_collections, dones=done_collections,
                                                        gamma=0.99)
                yield state_collections, action_collections, advantages, value_target
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
            value_collections[t % horizon] = self.agent.net.forward(state.to(device))[1]

            state = preprocess(observation)  # next state
            t += 1
            total_reward += reward


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    device = torch.device("cuda:0")
    LEARN = True
    i, episode = 0, 0
    n_state = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        discrete = False
        n_action = env.action_space.shape[0]
    else:
        discrete = True
        n_action = env.action_space.n
    ppo = PPO.Model(net=PPO.Net, learn=LEARN, device=device, n_state=n_state, n_action=n_action, discrete=discrete,
                    learning_rate=0.00025, epsilon=0.1)
    workers = [Worker(agent=ppo, worker_id=i) for i in range(4)]
    iter_units = []
    process_list = []
    for worker in workers:
        process = mp.Process(target=worker.work, args=(discrete,))
        process_list.append(process)
        iter_units.append(worker.work(discrete=discrete))
        process.start()

    for process in process_list:
        process.join()

    while True:
        s_list, a_list, adv_list, vt_list = [], [], [], []
        for index, iter_unit in enumerate(iter_units):
            s, a, ad, vt = iter_unit.__next__()
            loss = ppo.learn(state_collections=s, action_collections=a, advantage_collections=ad,
                             value_target=vt)
        #     s_list.append(s)
        #     a_list.append(a)
        #     adv_list.append(ad)
        #     vt_list.append(vt)
        # states, actions, adv, v_target = torch.cat(s_list), torch.cat(a_list), torch.cat(adv_list), torch.cat(vt_list)
        # loss = ppo.learn(state_collections=states, action_collections=actions, advantage_collections=adv,
        #                  value_target=v_target)
        # print('##############  Iteration: {}   ->    Loss: {}  ##############'.format(i, loss))
        i += 1
