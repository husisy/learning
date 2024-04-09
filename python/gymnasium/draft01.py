# Training an Agent
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns

plt.ion()

from utils import QLearningAgent

hf_moving_average = lambda x, w: np.convolve(x, np.ones(w), "same") / w


def plot_figure(player_count, dealer_count, value_grid, policy_grid, title):
    fig,(ax0,ax1) = plt.subplots(1,2,figsize=(12,4.8))
    fig.suptitle(title, fontsize=16)

    tmp0,tmp1 = np.meshgrid(player_count, dealer_count, indexing='ij')
    ax0.contourf(tmp0, tmp1, value_grid, cmap='viridis')
    ax0.set_yticks(dealer_count)
    ax0.set_ylabel("Dealer showing")
    ax0.set_yticklabels([('A' if x==1 else str(x)) for x in dealer_count])
    ax0.set_xticks(player_count)
    ax0.set_xlabel("Player sum")
    ax0.set_title(f"State values")

    ax1 = sns.heatmap(policy_grid.T, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax1.set_title(f"Policy")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.set_xticklabels(player_count)
    ax1.set_yticklabels([('A' if x==1 else str(x)) for x in dealer_count], fontsize=12)
    tmp0 = [
        matplotlib.patches.Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        matplotlib.patches.Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax1.legend(handles=tmp0, bbox_to_anchor=(1.3, 1))
    return fig


learning_rate = 0.01
n_episodes = 100000 #1000000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
agent = QLearningAgent(2, learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

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

# gym.wrappers.RecordEpisodeStatistics
env.time_queue
env.return_queue
env.length_queue

fig,ax = plt.subplots()
ax.plot(hf_moving_average(np.array(env.return_queue), 500), label='rewards')
ax.plot(hf_moving_average(np.array(env.length_queue), 500), label='lengths')
tmp0 = hf_moving_average(np.array(agent.training_error), 500)
ax.plot(np.linspace(0, n_episodes-1, len(tmp0)), tmp0, label='training error')
ax.legend()
fig.tight_layout()

player_count = list(range(12, 22))
dealer_count = list(range(1, 11))
tmp0 = itertools.product((0,1), player_count, dealer_count)
z0 = np.array([agent.q_values.get((p,d,a), (0,0)) for a,p,d in tmp0]).reshape(2, len(player_count), len(dealer_count), -1)
state_value_a01 = np.max(z0, axis=-1)
policy_a01 = np.argmax(z0, axis=-1)

fig0 = plot_figure(player_count, dealer_count, state_value_a01[0], policy_a01[0], title='Without usable ace')
fig1 = plot_figure(player_count, dealer_count, state_value_a01[1], policy_a01[1], title='With usable ace')
