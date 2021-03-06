import sources.PPO as PPO
import torch
import torch.nn as nn
import multiprocessing as mp
import gym


def preprocess(obs):
    return torch.from_numpy(obs).float()


class Worker:
    def __init__(self, worker_id, agent, episode):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.horizon = 32
        self.agent = agent
        self.episode = episode

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
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        episode = self.episode
        t, T, total_reward, horizon = 0, 0, 0, self.horizon
        done = True
        # reward_array = np.array([])

        observation = env.reset()
        state = preprocess(observation)

        state_collections = torch.zeros(horizon, env.observation_space.shape[0])
        action_collections = torch.zeros(horizon)
        reward_collections = torch.zeros(horizon)
        done_collections = torch.zeros(horizon + 1)
        while True:
            if done:
                observation = env.reset()
                state = preprocess(observation)
                print("Episode {}: finished after {} steps".format(episode.value, t - T))
                print("                        total_reward -> {}".format(total_reward))
                T = t
                with mp.Lock():
                    episode.value += 1
                total_reward = 0
            if LEARN is False:
                env.render()
            action = self.agent.choose_action(state.unsqueeze(0).to(device))
            if discrete:
                observation, reward, done, info = env.step(int(action))
            else:
                observation, reward, done, info = env.step([float(action)])

            if t > 0 and t % horizon == 0:
                done_collections[horizon] = done
                queue.put([state_collections, action_collections, done_collections, reward_collections, state])
                barrier.wait()
                state_collections = torch.zeros(horizon, env.observation_space.shape[0])
                action_collections = torch.zeros(horizon)
                reward_collections = torch.zeros(horizon)
                done_collections = torch.zeros(horizon + 1)
            state_collections[t % horizon] = state
            action_collections[t % horizon] = action
            reward_collections[t % horizon] = reward
            done_collections[t % horizon] = done

            state = preprocess(observation)  # next state
            t += 1
            total_reward += reward


if __name__ == '__main__':
    THREAD_NUMBER = 2
    env = gym.make('CartPole-v1')
    device = torch.device("cuda:0")
    mp.set_start_method('spawn', force=True)
    # device = torch.device('cpu')
    barrier = mp.Barrier(THREAD_NUMBER + 1)
    queue = mp.Queue()

    LEARN = True
    i = 0
    episode = mp.Value('i', 0)
    n_state = env.observation_space.shape[0]
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

    ppo = PPO.Model(net=net, old_net=old_net, learn=LEARN, device=device, n_state=n_state, n_action=n_action,
                    discrete=discrete, learning_rate=0.00025, epsilon=0.1)
    workers = [Worker(worker_id=i, agent=ppo, episode=episode) for i in range(THREAD_NUMBER)]
    process_list = []

    for worker in workers:
        process = mp.Process(target=worker.work, args=(discrete, barrier, queue))
        process_list.append(process)
        process.start()

    while i < 10000:
        for worker in workers:
            results = queue.get()
            s, a, d_collect, r_collect, final_s = results[0], results[1], results[2], results[3], results[4]
            ad, vt = worker.cal_adv(states=s, rewards=r_collect, dones=d_collect, final_state=final_s, gamma=0.99)
            agent_id, loss = worker.agent.learn(state_collections=s, action_collections=a, advantage_collections=ad,
                                                value_target=vt)
        barrier.wait()
        i += 1

    for process in process_list:
        process.join()
