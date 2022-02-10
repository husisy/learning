import os
import gym
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
import tianshou.utils.net.common

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    args = parser.parse_known_args()[0]
    return args


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
        obs = obs.to(self.model[0].weight.device)
        logits = self.model(obs.view(batch, -1))
        return logits, state


args = get_args()
args.device = 'cpu' #cuda

env = gym.make(args.task)
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
# train_envs = gym.make(args.task)
# you can also use tianshou.env.SubprocVectorEnv
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
# test_envs = gym.make(args.task)
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])

# np.random.seed(233)
# torch.manual_seed(233)
# train_envs.seed(233)
# test_envs.seed(233)

net = Net(args.state_shape, args.action_shape).to(args.device)
# dueling=(1, 1)
# net = ts.utils.net.common.Net(args.layer_num, args.state_shape, args.action_shape, args.device).to(args.device)
optim = torch.optim.Adam(net.parameters(), lr=args.lr)
policy = ts.policy.DQNPolicy(net, optim, args.gamma, args.n_step, target_update_freq=args.target_update_freq)

if args.prioritized_replay > 0:
    buf = ts.data.PrioritizedReplayBuffer(args.buffer_size, alpha=args.alpha, beta=args.beta)
else:
    buf = ts.data.ReplayBuffer(args.buffer_size)

train_collector = ts.data.Collector(policy, train_envs, buf)
test_collector = ts.data.Collector(policy, test_envs)

# policy.set_eps(1)
train_collector.collect(n_step=args.batch_size)

log_path = os.path.join(args.logdir, args.task, 'dqn')
writer = SummaryWriter(log_path)

def stop_fn(mean_rewards):
    return mean_rewards >= env.spec.reward_threshold

def train_fn(epoch, env_step):
    # eps annnealing
    eps_train = args.eps_train
    if env_step <= 10000:
        eps = eps_train
    elif env_step <= 50000:
        eps = eps_train * (1 - 0.9*(env_step-10000)/40000)
    else:
        eps = eps_train * 0.1
    policy.set_eps(eps)

def test_fn(epoch, env_step):
    policy.set_eps(args.eps_test)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, args.epoch,
    args.step_per_epoch, args.collect_per_step, args.test_num,
    args.batch_size, train_fn=train_fn, test_fn=test_fn,
    stop_fn=stop_fn, writer=writer)
# result(dict)
#   "train_step"(int) "train_episode"(float) "train_time/collector"(str) "train_time/model"(str) "train_speed"(str)
#   "test_step"(int) "test_episode"(float) "test_time"(str) "test_speed"(str)
#   "best_reward"(float) "best_result"(str) "duration"(str)

assert stop_fn(result['best_reward'])
print(result)
env = gym.make(args.task)
policy.eval()
policy.set_eps(args.eps_test)
collector = ts.data.Collector(policy, env)
result = collector.collect(n_episode=1, render=args.render)
print(f'Final reward: {result["rew"]}, length: {result["len"]}')

