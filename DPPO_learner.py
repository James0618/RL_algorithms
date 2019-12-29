import torch
from PPO import PolicyNet


def learn(self, state_collections, action_collections, reward_collections, final_state):
    """
    :param state_collections: Tensor[T, state_features]
    :param action_collections: Tensor[T]
    :param reward_collections: Tensor[T]
    :param final_state: Tensor[state_features]
    :return:
    """
    T = state_collections.shape[0]
    returns = self.state_value.forward(final_state)  # V(S_final)
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

    idx = [i for i in range(advantage_collections.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    advantage_collections = advantage_collections.index_select(0, idx)

    # torch.save(self.policy.state_dict(), 'params/old_policy_net.pkl')
    old_policy = PolicyNet(n_state=self.n_state, n_action=self.n_action)
    old_policy.load_state_dict(self.policy.state_dict())

    policy_optimizer = torch.optim.SGD(params=self.policy.parameters(), lr=self.learning_rate)
    state_value_optimizer = torch.optim.SGD(params=self.state_value.parameters(), lr=2*self.learning_rate)
    for j in range(12):
        # loss = mean(ratio * advantages) - lambda * KL(old_net, net)
        # J = -loss
        # print(torch.exp(self.policy.forward(state_collections).log_prob(action_collections)))
        ratio = torch.div(torch.exp(self.policy.forward(state_collections).log_prob(action_collections)),
                          torch.exp(old_policy.forward(state_collections).log_prob(action_collections)))
        policy_loss = - torch.mean(torch.min(torch.mul(ratio, advantage_collections.detach()),
                                             torch.mul(torch.clamp(ratio, min=1-self.epsilon, max=1+self.epsilon),
                                                       advantage_collections.detach())
                                             )
                                   )
        policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_optimizer.step()

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