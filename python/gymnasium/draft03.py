#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym

import torch

plt.ion()

Transition = collections.namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity, seed=None):
        self.memory = collections.deque([], maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        assert batch_size>=1
        transitions = self.rng.sample(self.memory, batch_size)
        non_final_mask = torch.tensor([(x.next_state is not None) for x in transitions], dtype=torch.bool)
        non_final_next_state = torch.stack([x.next_state for x in transitions if x.next_state is not None])
        state = torch.stack([x.state for x in transitions])
        action = torch.stack([x.action for x in transitions])
        reward = torch.stack([x.reward for x in transitions])
        return non_final_mask,non_final_next_state, state, action, reward


    def __len__(self):
        return len(self.memory)

class DQN(torch.nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def soft_update(target,  policy, tau):
    tmp0 = target.state_dict()
    tmp1 = policy.state_dict()
    for key in tmp1:
        tmp0[key] = tmp0[key]*(1-tau) + tmp1[key]*tau
    target.load_state_dict(tmp0)


batch_size = 128
discount_factor = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
num_episodes = 600


env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
n_actions = env.action_space.n #Discrete(2)
# Get the number of state observations
state, info = env.reset()
n_observations = len(state) #Box(4,)

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
memory = ReplayMemory(10000)

ind_step = 0
rng = random.Random()

rewards_list = []
with tqdm(range(num_episodes)) as pbar:
    for _ in pbar:
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        while True:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * ind_step / EPS_DECAY)
            ind_step += 1
            if rng.random() > eps_threshold:
                with torch.no_grad():
                    action = torch.argmax(policy_net(state.view(1,-1))[0])
            else:
                action = torch.tensor(env.action_space.sample(), dtype=torch.int64)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor(reward)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32)
            memory.push(state, action, next_state, reward)

            state = next_state
            if len(memory) >= batch_size:
                non_final_mask,non_final_next_states, state_batch, action_batch, reward_batch = memory.sample(batch_size)
                state_action_values = torch.gather(policy_net(state_batch), 1, action_batch.view(-1,1))[:,0]
                next_state_values = torch.zeros(batch_size) #value for final states is 0
                with torch.no_grad():
                    next_state_values[non_final_mask] = torch.max(target_net(non_final_next_states), 1).values
                tmp0 = (next_state_values * discount_factor) + reward_batch
                loss = torch.nn.functional.smooth_l1_loss(state_action_values, tmp0)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

                soft_update(target_net,  policy_net, TAU)
            if terminated or truncated:
                break
        rewards_list.append(env.return_queue[-1])
        pbar.set_description(f"Reward: {int(np.mean(rewards_list[-20:]))}")
env.close()


hf_moving_average = lambda x, w: np.convolve(x, np.ones(w), "same") / w
ydata = hf_moving_average(np.array(rewards_list), 50)
fig,ax = plt.subplots()
ax.plot(ydata)
fig.tight_layout()
