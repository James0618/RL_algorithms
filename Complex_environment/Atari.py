import sources.DQN as DQN
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gym


class Network(nn.Module):
    def __init__(self, output_shape):
        super(Network, self).__init__()
        self.conv = nn.Sequential(      # input_shape: 3*210*160
            nn.Conv2d(
                in_channels=6,          # RGB: 3 channels * 2 images
                groups=2,               # 1 state -> 2 images
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
    device = torch.device("cuda:0")
    net = net.to(device)
    agent = DQN.DQN(network=net, device=device, n_replay=1000, n_action=env.action_space.n, learning_rate=0.005,
                    learn=LEARN)
    observation = env.reset()
    observation = transforms.ToTensor()(observation).unsqueeze(0)
    env.reset()

    for episode in range(2000):
        observation = env.reset()
        observation = transforms.ToTensor()(observation).unsqueeze(0)
        state = observation
        reward = 0
        for t in range(1000):
            if episode > 200 or LEARN is False:
                env.render()
            if t < 1:
                observation, _, done, info = env.step(0)
                observation = transforms.ToTensor()(observation).unsqueeze(0)   # 1*3*210*160
                state = torch.cat((state, observation), dim=1)                  # 1*6*210*160
            else:
                state_before = state
                action = agent.choose_action(state)
                # print('action: {}'.format(action))
                observation, reward, done, info = env.step(int(action))
                observation = transforms.ToTensor()(observation).unsqueeze(0)
                indices = torch.LongTensor([3, 4, 5])
                state = torch.cat((torch.index_select(state, 1, indices), observation), dim=1)  # generate next state
                if t == 500 - 1:
                    done = True

                transition = [state_before, action, reward, state]
                agent.store_transition(transition=transition)
                if done:
                    print("Episode {}: Reward -> {}".format(episode, reward))
                    break

                # learn when replay has enough transitions
                if episode >= 3:
                    if t % 50 == 0 and LEARN is True:
                        pass
                        # agent.learn()

                # save success params
                if t > 400:
                    if episode % 20 == 0:
                        agent.save_net()
