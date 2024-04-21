import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
import gymnasium as gym
from tqdm import tqdm
# pip install swig mujoco
plt.ion()


def np_cumulative_discount(rewards:np.ndarray, discount:float):
    """
    C[i] = R[i] + discount * C[i+1]
    signal.lfilter(b, a, x, axis=-1, zi=None)
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M] - a[1]*y[n-1] - ... - a[N]*y[n-N]
    """
    # https://stackoverflow.com/a/47971187/7290857
    # https://stackoverflow.com/q/77752955/7290857
    r = rewards[::-1]
    a = [1, -discount]
    b = [1]
    ret = np.ascontiguousarray(scipy.signal.lfilter(b, a, x=r)[::-1])
    return ret


class ReinforceAlgorithm(torch.nn.Module):
    def __init__(self, obs_space_dims, action_space_dims):
        super().__init__()
        self.gamma = 0.99
        self.eps = 1e-6  # small number for mathematical stability
        tmp0 = [obs_space_dims, 16, 32, 2*action_space_dims]
        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(x,y) for x,y in zip(tmp0[:-1], tmp0[1:])])
        self.probs = []
        self.rewards = []

    def sample_action(self, state):
        x = torch.tensor(np.array([state]), dtype=torch.float32)
        for fc in self.fc_list[:-1]:
            x = torch.nn.functional.leaky_relu(fc(x))
        x = self.fc_list[-1](x)
        dim_action_space = x.shape[-1]//2
        action_means = x[:, :dim_action_space]
        action_stddevs = torch.nn.functional.softplus(x[:, dim_action_space:])
        distrib = torch.distributions.Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        self.probs.append(distrib.log_prob(action)[0])
        return action.numpy()

    def forward(self, tag_clear=True):
        log_prob_mean = torch.stack(self.probs)
        tmp0 = np_cumulative_discount(np.array(self.rewards), self.gamma)
        loss = - torch.mean(log_prob_mean * torch.tensor(tmp0, dtype=log_prob_mean.dtype))
        # loss = -torch.dot(log_prob_mean, torch.tensor(tmp0, dtype=log_prob_mean.dtype))
        if tag_clear:
            self.probs.clear()
            self.rewards.clear()
        return loss


env = gym.make("InvertedPendulum-v5")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=50)

num_episode = 1500 #about 2 minutes
obs_space_dims = env.observation_space.shape[0] #4
action_space_dims = env.action_space.shape[0] #1
agent = ReinforceAlgorithm(obs_space_dims, action_space_dims)

# lr_init = 0.003
# lr_end = 1e-4
# optimizer = torch.optim.AdamW(agent.parameters(), lr=0.001)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
# tmp0 = (lr_end/lr_init)**(1/num_episode)
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=tmp0)

rewards_list = []
with tqdm(range(num_episode)) as pbar:
    for episode in pbar:
        obs, info = wrapped_env.reset()
        while True:
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            if terminated or truncated:
                break

        rewards_list.append(wrapped_env.return_queue[-1])
        optimizer.zero_grad()
        loss = agent()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Reward: {int(np.mean(rewards_list[-20:]))}")
        # lr_scheduler.step()
env.close()

hf_moving_average = lambda x, w: np.convolve(x, np.ones(w), "same") / w
ydata = hf_moving_average(np.array(rewards_list), 50)
fig,ax = plt.subplots()
ax.plot(ydata)
fig.tight_layout()
