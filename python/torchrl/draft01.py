#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import torch

plt.ion()

np_rng = np.random.default_rng()


class ReplayMemory:
    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.memory = []
        self.np_rng = np.random.default_rng(seed)

    def push(self, state, action, next_state, reward):
        if len(self.memory) >= self.capacity:
            for _ in range(1+len(self.memory)-self.capacity):
                self.memory.pop(0)
        self.memory.append(dict(state=state, action=action, next_state=next_state, reward=reward))

    def sample(self, batch_size):
        assert batch_size>=1
        index = np.sort(self.np_rng.choice(len(self.memory), batch_size, replace=False, shuffle=False))
        transitions = [self.memory[i] for i in index]
        non_final_mask = torch.tensor([(x['next_state'] is not None) for x in transitions], dtype=torch.bool)
        non_final_next_state = torch.stack([x['next_state'] for x in transitions if x['next_state'] is not None])
        state = torch.stack([x['state'] for x in transitions])
        action = torch.stack([x['action'] for x in transitions])
        reward = torch.stack([x['reward'] for x in transitions])
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

class ExponentialDecay:
    def __init__(self, start, end, decay):
        self.ind_step = 0
        self.start = float(start)
        self.end = float(end)
        self.decay = float(decay)

    def __call__(self, step=True):
        ret = self.end + (self.start - self.end) * np.exp(-self.ind_step / self.decay)
        if step:
            self.ind_step += 1
        return ret


batch_size = 128
discount_factor = 0.99
num_episodes = 600
epsilon = ExponentialDecay(start=0.9, end=0.05, decay=1000)

env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
n_actions = env.action_space.n #Discrete(2)
n_observations = env.observation_space.shape[0] #4
policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
memory = ReplayMemory(10000)


rewards_list = []
with tqdm(range(num_episodes)) as pbar:
    for _ in pbar:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        while True:
            if np_rng.uniform(0,1) > epsilon(step=True):
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

                soft_update(target_net,  policy_net, tau=0.005)
            if terminated or truncated:
                break
        rewards_list.append(env.return_queue[-1])
        pbar.set_description(f"Reward: {int(np.mean(rewards_list[-20:]))}")
env.close()


hf_moving_average = lambda x, w: np.convolve(x, np.ones(w), "same") / w
ydata = hf_moving_average(np.array(rewards_list), 10)
fig,ax = plt.subplots()
ax.plot(ydata)
fig.tight_layout()
