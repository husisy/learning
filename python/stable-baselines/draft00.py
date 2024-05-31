import os
import numpy as np
import gymnasium as gym
import stable_baselines3

from zzz233 import next_tbd_dir

# from stable_baselines3 import PPO
# from stable_baselines3.ppo.policies import MlpPolicy
# from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
# from stable_baselines3 import A2C, SAC, PPO, TD3


env = gym.make("CartPole-v1", render_mode="rgb_array")
model = stable_baselines3.A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #     obs = vec_env.reset()


## https://github.com/araffin/rl-tutorial-jnrr19 1-getting-started
def evaluate(model:stable_baselines3.common.base_class.BaseAlgorithm, num_episodes:int=100, deterministic:bool=True) -> float:
    vec_env = model.get_env()
    rewards_list = [[] for _ in range(num_episodes)]
    obs = vec_env.reset()
    for ind0 in range(num_episodes):
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _info = vec_env.step(action)
            rewards_list[ind0].append(reward.item())
    mean_episode_reward = np.array([sum(x) for x in rewards_list]).mean().item()
    return mean_episode_reward

env = gym.make("CartPole-v1")
model = stable_baselines3.PPO(stable_baselines3.ppo.policies.MlpPolicy, env)
mean_reward_before_train = evaluate(model, num_episodes=100, deterministic=True)
print(f"Mean reward: {mean_reward_before_train:.2f}")
mean_reward, std_reward = stable_baselines3.common.evaluation.evaluate_policy(model, env, n_eval_episodes=100) #warn=False
print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

model.learn(total_timesteps=10_000)
mean_reward, std_reward = stable_baselines3.common.evaluation.evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
## train model in one line
# model = stable_baselines3.PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)

## record video
hf0 = lambda: gym.make("CartPole-v1", render_mode="rgb_array")
eval_env = stable_baselines3.common.vec_env.DummyVecEnv([hf0])
logdir = next_tbd_dir(tag_create=False)
# Start the video at step=0 and record 500 steps
video_length = 500
hf1 = lambda x: x==0
eval_env = stable_baselines3.common.vec_env.VecVideoRecorder(eval_env, logdir,
            record_video_trigger=hf1, video_length=video_length, name_prefix="ppo-cartpole")
obs = eval_env.reset()
for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)
eval_env.close()


## https://github.com/araffin/rl-tutorial-jnrr19 2-gym-wrappers-saving-loading
save_dir = next_tbd_dir()
hf_logdir = lambda *x: os.path.join(save_dir, *x)

model = stable_baselines3.PPO("MlpPolicy", "Pendulum-v1").learn(8000)
model.save(hf_logdir('PPO_tutorial')) #PPO_tutorial.zip
obs = model.env.observation_space.sample()
action,_ = model.predict(obs, deterministic=True) #(np, float32, (1,)
model1 = stable_baselines3.PPO.load(hf_logdir('PPO_tutorial'))
action1,_ = model1.predict(obs, deterministic=True)
assert np.abs(action-action1).max().item() < 1e-6
'''
saved_model.zip/
├── data              JSON file of class-parameters (dictionary)
├── parameter_list    JSON file of model parameters and their ordering (list)
├── parameters        Bytes from numpy.savez (a zip file of the numpy arrays). ...
    ├── ...           Being a zip-archive itself, this object can also be opened ...
        ├── ...       as a zip-archive and browsed.
'''

model = stable_baselines3.A2C("MlpPolicy", "Pendulum-v1", gamma=0.9, n_steps=20).learn(8000)
model.save(hf_logdir("A2C_tutorial"))
model1 = stable_baselines3.A2C.load(f"{save_dir}/A2C_tutorial", verbose=1)
model1.gamma
model1.n_steps
# as the environment is not serializable, we need to set a new instance of the environment
model1.set_env(stable_baselines3.common.vec_env.DummyVecEnv([lambda: gym.make("Pendulum-v1")]))
model1.learn(8_000) #continue training


# https://gymnasium.farama.org/api/wrappers/
# gym.wrappers.TimeLimit
class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_steps:int):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

# Here we create the environment directly because gym.make() already wrap the environment in a TimeLimit wrapper otherwise
env = gym.envs.classic_control.PendulumEnv()
env = TimeLimitWrapper(env, max_steps=100)
obs, _ = env.reset()
n_steps = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    n_steps += 1
    if terminated or truncated:
        break


class NormalizeActionWrapper(gym.Wrapper):
    def __init__(self, env):
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        self.low, self.high = action_space.low, action_space.high
        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.low + 0.5 * (self.high - self.low) * (action + 1.0)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info

env0 = gym.make("Pendulum-v1")
env0.action_space.low, env0.action_space.high #[-2,2]
print(np.array([env0.action_space.sample().item() for _ in range(10)]))
env1 = NormalizeActionWrapper(gym.make("Pendulum-v1"))
print(np.array([env1.action_space.sample().item() for _ in range(10)]))

hf0 = lambda: stable_baselines3.common.monitor.Monitor(gym.make("Pendulum-v1"))
env = stable_baselines3.common.vec_env.DummyVecEnv([hf0])
model = stable_baselines3.A2C("MlpPolicy", env, verbose=1).learn(1000)
# stable_baselines3.common.evaluation.evaluate_policy(model, hf0(), n_eval_episodes=100)
hf0 = lambda: NormalizeActionWrapper(stable_baselines3.common.monitor.Monitor(gym.make("Pendulum-v1")))
normalized_env = stable_baselines3.common.vec_env.DummyVecEnv([hf0])
model1 = stable_baselines3.A2C("MlpPolicy", normalized_env, verbose=1).learn(1000)


# VecEnvWrappers
# from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
hf0 = lambda: gym.make("Pendulum-v1")
env = stable_baselines3.common.vec_env.DummyVecEnv([hf0])
normalized_vec_env = stable_baselines3.common.vec_env.VecNormalize(env)
obs = normalized_vec_env.reset()
for _ in range(10):
    action = normalized_vec_env.action_space.sample().reshape(env.num_envs, -1)
    obs, reward, _, _ = normalized_vec_env.step(action)
    print(obs, reward)
