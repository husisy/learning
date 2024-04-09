# create a custom environment
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

from utils import QLearningAgent

hf_moving_average = lambda x, w: np.convolve(x, np.ones(w), "same") / w


class GridWorldEnv(gym.Env):
    def __init__(self, size:int=5, tag_fix_target:bool=False):
        self.dtype = np.int64
        self.size = size
        self._agent_location = np.array([-1,-1], dtype=self.dtype)
        self._target_location = np.array([-1,-1], dtype=self.dtype)
        tmp0 = gym.spaces.Box(0, size-1, shape=(2,), dtype=int)
        self.observation_space = gym.spaces.Dict({"agent":tmp0, "target":tmp0})
        self.tag_fix_target = tag_fix_target

        tmp0 = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)} #right, up, left, down
        self.action_space = gym.spaces.Discrete(len(tmp0))
        self._action_to_direction = {k:np.array(v, dtype=self.dtype) for k,v in tmp0.items()}

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        tmp0 = np.abs(self._agent_location-self._target_location).sum() #L1-norm
        return {"distance": tmp0}

    def reset(self, seed:(int|None)=None, options:(dict|None)=None):
        super().reset(seed=seed)
        if (self._target_location[0]==-1) or (not self.tag_fix_target):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=self.dtype)
        tmp0 = self._target_location
        while np.all(tmp0==self._target_location):
            tmp0 = self.np_random.integers(0, self.size, size=2, dtype=self.dtype)
        self._agent_location = tmp0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        tmp0 = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location+tmp0, 0, self.size-1)
        terminated = np.all(self._target_location==self._agent_location)
        truncated = False
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)
# gym.pprint_registry()

gym.make("gymnasium_env/GridWorld-v0")
gym.make("gymnasium_env/GridWorld-v0", max_episode_steps=100)
gym.make("gymnasium_env/GridWorld-v0", size=10)
gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)


size = 5
learning_rate = 0.01
n_episodes = 3000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.wrappers.FlattenObservation(gym.make("gymnasium_env/GridWorld-v0", size=size, tag_fix_target=True))
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
agent = QLearningAgent(4, learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    while True:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)
        obs = next_obs
        if terminated or truncated:
            break
    agent.decay_epsilon()

target = list(agent.q_values.keys())[0][2:]
tmp0 = {k[:2]:v for k,v in agent.q_values.items()}
tmp1 = {0:'→', 1:'↑', 2:'←', 3:'↓'}
z0 = [[tmp1[np.argmax(tmp0.get((x,y), np.zeros(4)))] for y in range(size)] for x in range(size)]
z0[target[0]][target[1]] = 'T'
print(np.array(z0)[:,::-1].T)

fig,ax = plt.subplots()
ax.plot(hf_moving_average(np.array(env.length_queue), 500), label='lengths')
ax.legend()
fig.tight_layout()
