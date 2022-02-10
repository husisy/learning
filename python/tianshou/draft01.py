import gym
import numpy as np
import torch
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('tbd00/dqn')

env = gym.make('CartPole-v0')
train_envs = gym.make('CartPole-v0')
test_envs = gym.make('CartPole-v0')

train_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(8)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])


class Net(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = torch.nn.Sequential(*[
            torch.nn.Linear(np.prod(state_shape), 128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, np.prod(action_shape))
        ])
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
test_collector = ts.data.Collector(policy, test_envs)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=1000, collect_per_step=10,
    episode_per_test=100, batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    writer=writer)
print(f'Finished training! Use {result["duration"]}')


policy.eval()
policy.set_eps(0.05)
# test_collector.reset()
# test_collector.collect(n_episode=1)

for _ in range(10):
    env = gym.make('CartPole-v0')
    data = ts.data.Batch(state=ts.data.Batch(), obs={}, act={}, rew={},
            done={}, info={}, obs_next=ts.data.Batch(), policy=ts.data.Batch())
    with torch.no_grad():
        data.obs = [env.reset()]
        total_reward = 0
        for t in range(300):
            action = policy(data)['act'][0].item()
            obs, reward, done, info = env.step(action)
            total_reward = total_reward + reward
            data.obs = [obs]
            if done:
                break
    print(f'total reward: {total_reward}') #most of time is 200
