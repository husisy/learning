import time
import numpy as np
import gym


# play animation in jupyter notebook connecting remote server
# https://stackoverflow.com/a/44426542/7290857

# CartPole-v0
# MountainCar-v0
# MsPacman-v0
# Hopper-v1 #fail on windows
env = gym.make('CartPole-v1', render_mode="rgb_array")
for i_episode in range(3):
    observation = env.reset()
    for t in range(300):
        env.render()
        # print(observation) #numpy array
        action = env.action_space.sample() #take a random action
        # action(int) {0,1}
        #   action=0: left
        #   action=1: right
        observation, reward, terminated, done, info = env.step(action)
        # observation/state(np,float,(4,)): position, velocity, angle, angular velocity
        # reward(float)
        # done(bool)
        # info(dict)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


env = gym.make('CartPole-v0')
env.action_space
env.observation_space
env.observation_space.high
env.observation_space.low


z0 = gym.spaces.Discrete(8) #{0,1,2,3,4,5,6,7}
x = z0.sample()
assert z0.contains(x)
z0.n #8


z0 = []
env = gym.make('CartPole-v0')
observation = env.reset()
for t in range(300):
    env.render()
    action = env.action_space.sample() #take a random action
    # action = np.random.rand()>0.7 #most are 0
    # action=0: apply force from right
    # action=1: apply force from left
    observation, reward, done, info = env.step(action)
    # reward are always 1.0, otherwise done (moves more then 2.4 units away from center)
    z0.append(reward)
    time.sleep(0.01)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
set(z0)

gym.envs.registry.all() #859@20200405
