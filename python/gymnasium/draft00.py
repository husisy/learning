import logging
import numpy as np
import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake

from utils import next_tbd_dir

# CartPole-v1
env = gym.make('CartPole-v1', render_mode="human")
env.action_space
env.observation_space
env.observation_space.high
env.observation_space.low
for i_episode in range(3):
    observation = env.reset()
    for t in range(300):
        action = env.action_space.sample() #take a random action
        # action(int) {0,1}
        #   action=0: left
        #   action=1: right
        observation, reward, terminated, truncated, info = env.step(action)
        # observation/state(np,float,(4,)): position, velocity, angle, angular velocity
        # reward(float)
        # truncated(bool)
        # info(dict)
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


env = gym.make("LunarLander-v2", render_mode="human")
env.action_space
env.observation_space
observation, info = env.reset(seed=42)
for _ in range(200):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()


# Show the initial state
tmp0 = gymnasium.envs.toy_text.frozen_lake.generate_random_map(size=4, p=0.9)
env = gym.make("FrozenLake-v1", render_mode="human", desc=tmp0)
env.reset()
# env.render()



z0 = gym.spaces.Discrete(8) #{0,1,2,3,4,5,6,7}
x = z0.sample()
assert z0.contains(x)
z0.n #8


gym.pprint_registry()
'''
===== classic_control =====
Acrobot-v1             CartPole-v0            CartPole-v1
MountainCar-v0         MountainCarContinuous-v0 Pendulum-v1
===== phys2d =====
phys2d/CartPole-v0     phys2d/CartPole-v1     phys2d/Pendulum-v0
===== box2d =====
BipedalWalker-v3       BipedalWalkerHardcore-v3 CarRacing-v2
LunarLander-v2         LunarLanderContinuous-v2
===== toy_text =====
Blackjack-v1           CliffWalking-v0        FrozenLake-v1
FrozenLake8x8-v1       Taxi-v3
===== tabular =====
tabular/Blackjack-v0   tabular/CliffWalking-v0
===== mujoco =====
Ant-v2                 Ant-v3                 Ant-v4
Ant-v5                 HalfCheetah-v2         HalfCheetah-v3
HalfCheetah-v4         HalfCheetah-v5         Hopper-v2
Hopper-v3              Hopper-v4              Hopper-v5
Humanoid-v2            Humanoid-v3            Humanoid-v4
Humanoid-v5            HumanoidStandup-v2     HumanoidStandup-v4
HumanoidStandup-v5     InvertedDoublePendulum-v2 InvertedDoublePendulum-v4
InvertedDoublePendulum-v5 InvertedPendulum-v2    InvertedPendulum-v4
InvertedPendulum-v5    Pusher-v2              Pusher-v4
Pusher-v5              Reacher-v2             Reacher-v4
Reacher-v5             Swimmer-v2             Swimmer-v3
Swimmer-v4             Swimmer-v5             Walker2d-v2
Walker2d-v3            Walker2d-v4            Walker2d-v5
===== None =====
GymV21Environment-v0   GymV26Environment-v0
'''


def demo_recording_every_episode():
    # pip install moviepy
    # conda install -c conda-forge moviepy ffmpeg
    num_logging_period = 5
    num_episode = 20
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    logdir = next_tbd_dir(tag_create=False)
    hf0 = lambda x: x % num_logging_period == 0
    env = gym.wrappers.RecordVideo(env, video_folder=logdir, name_prefix="training", episode_trigger=hf0)
    env = gym.wrappers.RecordEpisodeStatistics(env) #buffer_length=4
    for ind_episode in range(num_episode):
        obs, info = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        logging.info(f"episode-{ind_episode}", info["episode"])
    env.close()
