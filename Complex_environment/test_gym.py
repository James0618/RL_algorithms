import common.atari_wrappers as wrappers
import gym


if __name__ == '__main__':
    # env = gym.make("SpaceInvaders-v0")
    env = wrappers.make_atari("SpaceInvaders-v0", 1000)
    env.reset()
    for episode in range(1000):
        env.reset()
        for t in range(1000):
            env.render()
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                print("episode {} ends after {} steps.".format(episode+1, t+1))
                print("         left live: {}".format(info))
                break
