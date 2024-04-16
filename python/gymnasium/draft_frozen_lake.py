# create a custom environment
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake

from utils import QLearningAgent
plt.ion()

hf_moving_average = lambda x, w: np.convolve(x, np.ones(w), "same") / w


size = 7
learning_rate = 0.05 #0.01
n_episodes = 4000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
proba_frozen = 0.9

env_sub = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode="rgb_array",
    desc=gymnasium.envs.toy_text.frozen_lake.generate_random_map(size=size, p=proba_frozen),
)
env = gym.wrappers.RecordEpisodeStatistics(env_sub, buffer_length=n_episodes)
agent = QLearningAgent(4, learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

obs_list = []
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    while True:
        action = agent.get_action((obs,))
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update((obs,), action, reward, terminated, (next_obs,))
        obs = next_obs
        if terminated or truncated:
            break
    agent.decay_epsilon()
env_image = env.render()
env.close()

qvalue_dict = {k[0]:v for k,v in agent.q_values.items()}
direction_dict = {0: "←", 1: "↓", 2: "→", 3: "↑"}
index_HG = [x for x in range(size*size) if env_sub.spec.kwargs['desc'][x//size][x%size] in 'HG']
# hole, goal
qvalue = np.array([np.max(qvalue_dict.get(x, np.zeros(4))) for x in range(size*size)])
qvalue[index_HG] = 0
action_map = [direction_dict[np.argmax(qvalue_dict.get(x, np.zeros(4)))] for x in range(size*size)]
for x in index_HG:
    action_map[x] = ''

fig,ax = plt.subplots()
ax.plot(hf_moving_average(np.array(env.length_queue), 50), label='lengths')
ax.set_xlabel('Episode')
ax.set_ylabel('Length')

fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(9, 4))
ax0.imshow(env_image)
himage = ax1.imshow(qvalue.reshape(size,size), cmap='Blues')
fig.colorbar(himage, ax=ax1)
for x in range(size*size):
    ax1.text(x%size, x//size, action_map[x], ha='center', va='center', color='black')
