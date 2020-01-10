import sources.DQN as DQN
import torch
import torch.nn as nn
import gym


class Network(nn.Module):
    def __init__(self, output_shape):
        super(Network, self).__init__()
        self.conv = nn.Sequential(      # input_shape: 3*210*160
            nn.Conv2d(
                in_channels=3,          # RGB: 3 channels
                out_channels=16,        # filters' number
                kernel_size=5,          # kernel's size
                stride=2,
                padding=2
            ),                          # output_shape: 16*105*80
            nn.ReLU(),
            nn.MaxPool2d(               # input_shape:  16*105*80
                kernel_size=3,
                stride=1,
                padding=1,
            ),                          # output_shape: 16*105*80
            nn.Conv2d(                  # input_shape:  16*105*80
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),                          # output_shape: 32*53*40
            nn.ReLU(),
            nn.MaxPool2d(               # input_shape:  32*53*40
                kernel_size=3,
                stride=1,
                padding=1,
            ),                          # output_shape: 32*53*40
            nn.Conv2d(                  # input_shape:  32*53*40
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2
            ),                          # output_shape: 64*27*20
            nn.ReLU(),
            nn.Conv2d(                  # input_shape:  64*27*20
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),                          # output_shape: 64*14*10
            nn.ReLU()
        )
        self.fcn = nn.Sequential(
            nn.Linear(64*14*10, 64*14*2),
            nn.ReLU(),
            nn.Linear(64*14*2, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape),
        )

    def forward(self, state):
        output = self.conv(state)
        output = output.view(output.size(0), -1)
        output = self.fcn(output)
        return output


if __name__ == '__main__':
    env = gym.make("SpaceInvaders-v0")
    LEARN = True
    net = Network(env.action_space.n)
    agent = DQN.DQN(network=net, n_replay=5000, n_action=2, n_state=4, learning_rate=0.005, learn=LEARN)
    observation = env.reset()
    observation = torch.FloatTensor(observation)
    observation = torch.transpose(observation, 0, 2)
    observation = torch.transpose(observation, 1, 2).unsqueeze(0)
    env.reset()

    for episode in range(2000):
        observation = env.reset()
        observation = torch.FloatTensor(observation)
        observation = torch.transpose(observation, 0, 2)
        observation = torch.transpose(observation, 1, 2).unsqueeze(0)
        state = observation
        reward = 0
        for t in range(500):
            if episode > 5000 or LEARN is False:
                env.render()
            if t < 1:
                observation, _, done, info = env.step(0)
                observation = torch.FloatTensor(observation)                    # 210*160*3
                observation = torch.transpose(observation, 0, 2)
                observation = torch.transpose(observation, 1, 2).unsqueeze(0)   # 1*3*210*160
                state = torch.cat((state, observation), dim=1)                  # 1*6*210*160
            else:
                state_before = state
                action = agent.choose_action(state)
                # print('action: {}'.format(action))
                observation, _, done, info = env.step(int(action))
                observation = torch.FloatTensor(observation)  # 210*160*3
                observation = torch.transpose(observation, 0, 2)
                observation = torch.transpose(observation, 1, 2).unsqueeze(0)  # 1*3*210*160
                if done:
                    reward = 0
                else:
                    reward = 1
                indices = torch.LongTensor([3, 4, 5])
                state = torch.cat((torch.index_select(state, 1, indices), observation), dim=1)  # next state
                if t == 500 - 1:
                    done = True

                if done:
                    print("Episode {}: finished after {} timesteps".format(episode, t + 1))
                    break
                transition = []
                transition.append(state_before)
                agent.store_transition(state.tolist() + [action] + [reward] + state_before.tolist())

                # learn when replay has enough transitions
                if episode >= 5:
                    if t % 3 == 0 and LEARN is True:
                        agent.learn()

                # save success params
                if t > 400:
                    if episode % 20 == 0:
                        agent.save_net()