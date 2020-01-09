import torch
import torch.multiprocessing as mp
from PPO import PolicyNet, ValueNet


class Agent(mp.Process):
    def __init__(self, policy, n_state, n_action, queue, global_episode, learn=True,
                 gamma=0.95, learning_rate=0.005, epsilon=0.1):
        super(Agent, self).__init__()

        if learn is True:
            self.policy = policy
            self.state_value = ValueNet(n_state=n_state)
        else:
            self.policy = policy
            self.load_net()
        self.old_policy = PolicyNet(n_state=n_state, n_action=n_action)

        self.queue = queue
        self.global_episode = global_episode
        self.MAX_EPISODE = 30000

        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.LEARN = learn

    def learn(self, state_collections, action_collections, reward_collections, final_state):
        T = state_collections.shape[0]

        returns = self.state_value.forward(final_state)  # V(S_final)
        advantage_collections = torch.FloatTensor([])  # Tensor[T]

        # calculate advantages
        for i in range(1, T + 1):
            # t: T -> 1
            # Returns[t] = r[t] + gamma * Returns[t+1]
            returns = reward_collections[T - i] + self.gamma * returns
            advantage = returns - self.state_value.forward(state_collections[T - i])
            if advantage_collections.shape[0] == 0:
                advantage_collections = advantage.unsqueeze(0)
            else:
                advantage_collections = torch.cat((advantage_collections, advantage.unsqueeze(0)))

        # reverse advantage_collections
        idx = [i for i in range(advantage_collections.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        advantage_collections = advantage_collections.index_select(0, idx)

        # torch.save(self.policy.state_dict(), 'params/old_policy_net.pkl')
        old_policy = PolicyNet(n_state=self.n_state, n_action=self.n_action)
        old_policy.load_state_dict(self.policy.state_dict())

        policy_optimizer = torch.optim.SGD(params=self.policy.parameters(), lr=self.learning_rate)
        state_value_optimizer = torch.optim.SGD(params=self.state_value.parameters(), lr=2*self.learning_rate)

        # update policy net
        for j in range(12):
            # loss = mean(ratio * advantages) - lambda * KL(old_net, net)
            # J = -loss
            # print(torch.exp(self.policy.forward(state_collections).log_prob(action_collections)))
            ratio = torch.div(torch.exp(self.policy.forward(state_collections).log_prob(action_collections)),
                              torch.exp(old_policy.forward(state_collections).log_prob(action_collections)))
            policy_loss = - torch.mean(torch.min(torch.mul(ratio, advantage_collections.detach()),
                                                 torch.mul(
                                                     torch.clamp(ratio, min=1 - self.epsilon, max=1 + self.epsilon),
                                                     advantage_collections.detach())
                                                 )
                                       )
            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

        # update state_value net
        for j in range(16):
            returns = self.state_value.forward(final_state)
            advantage_collections = torch.FloatTensor([])  # Tensor[T]
            for i in range(1, T + 1):
                # t: T -> 1
                # Returns[t] = r[t] + gamma * Returns[t+1]
                returns = reward_collections[T - i] + self.gamma * returns
                advantage = returns - self.state_value.forward(state_collections[T - i])
                if advantage_collections.shape[0] == 0:
                    advantage_collections = advantage.unsqueeze(0)
                else:
                    advantage_collections = torch.cat((advantage_collections, advantage.unsqueeze(0)))
            state_loss = torch.mean(torch.pow(advantage_collections, 2))
            state_value_optimizer.zero_grad()
            state_loss.backward(retain_graph=True)
            state_value_optimizer.step()

    def run(self):
        while self.global_episode.value < self.MAX_EPISODE:
            collections = self.queue.get()
            state_collections = collections[0]
            action_collections = collections[1]
            reward_collections = collections[2]
            final_state = collections[3]
            if self.LEARN is True:
                self.learn(state_collections=state_collections, action_collections=action_collections,
                           reward_collections=reward_collections, final_state=final_state)

    def load_net(self):
        policy = torch.load('params/ppo_policy.pkl')
        self.policy.load_state_dict(policy.state_dict())
        self.state_value = torch.load('params/ppo_state_value.pkl')


if __name__ == '__main__':
    print("This module could not be used directly.")
