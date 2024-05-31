import time
import numpy as np
import torch
import torchrl
import tensordict
import gymnasium as gym

from zzz233 import next_tbd_dir

# https://pytorch.org/rl/tutorials/getting-started-0.html
env = torchrl.envs.GymEnv("Pendulum-v1", device=None)
td0 = env.reset()
# td0: tensordict
#   done: (torch, bool, (1,))
#   observation: (torch, float32, (3,))
#   terminated: (torch, bool, (1,))
#   truncated: (torch, bool, (1,))
td0 = env.rand_action(td0) #in-place update
# td0: tensordict
#   action: (torch, float32, (1,))
td0 = env.step(td0) #in-place update
# td0: tensordict
#   next: tensordict
#     done: (torch, bool, (1,))
#     observation: (torch, float32, (3,))
#     reward: (torch, float32, (1,))
#     terminated: (torch, bool, (1,))
#     truncated: (torch, bool, (1,))
td1 = torchrl.envs.step_mdp(td0) #not in-place
# td1 <- td0['next']
# td1: tensordict
#   done: (torch, bool, (1,))
#   observation: (torch, float32, (3,))
#   terminated: (torch, bool, (1,))
#   truncated: (torch, bool, (1,))
assert torch.max(torch.abs(td1['observation'] - td0['next', 'observation'])).item() < 1e-6
td0 = env.rollout(max_steps=10)
td0.batch_size #10

env01 = torchrl.envs.TransformedEnv(env, torchrl.envs.StepCounter(max_steps=4))
td0 = env01.rollout(max_steps=5)
# td0: tensordict (as before)
#   step_count: (torch, int64, (10,1))
td0.batch_size #4
td0['next','done'] #[False, False, False, True]
td0['next','truncated'] #[False, False, False, True]
## rollout stop when done

# https://pytorch.org/rl/tutorials/getting-started-1.html
env = torchrl.envs.GymEnv("Pendulum-v1", device=None)
dim_observation = env.observation_spec['observation'].shape[-1]
dim_action = env.action_spec.shape[-1]
module = torch.nn.Linear(in_features=dim_observation, out_features=dim_action)
# module = torchrl.modules.MLP(out_features=dim_action, num_cells=[32, 64], activation_class=torch.nn.Tanh)
policy = tensordict.nn.TensorDictModule(module, in_keys=["observation"], out_keys=["action"])
# policy = torchrl.modules.Actor(module)
td0 = env.rollout(max_steps=10, policy=policy)

# probabilistic policy
env = torchrl.envs.GymEnv("Pendulum-v1", device=None)
module = torch.nn.Sequential(
    torchrl.modules.MLP(in_features=3, out_features=2),
    tensordict.nn.distributions.NormalParamExtractor(),
)
td_module = tensordict.nn.TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
policy = torchrl.modules.ProbabilisticActor(td_module, in_keys=["loc","scale"], out_keys=["action"],
    distribution_class=torch.distributions.Normal, return_log_prob=True)
# default_interaction_type
td0 = env.rollout(max_steps=10, policy=policy)
# td0: tensordict
#   sample_log_prob: (torch, float32, (10,1))
with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.MEAN):
    # takes the mean as action
    td0 = env.rollout(max_steps=10, policy=policy)
with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.RANDOM):
    # Samples actions according to the dist
    td0 = env.rollout(max_steps=10, policy=policy)
# default_interaction_type



policy = torchrl.modules.Actor(torchrl.modules.MLP(3, 1, num_cells=[32, 64]))
exploration_module = torchrl.modules.EGreedyModule(spec=env.action_spec, annealing_num_steps=1000, eps_init=0.5)
# eps=1 (random) -> eps=0 (deterministic)
exploration_policy = tensordict.nn.TensorDictSequential(policy, exploration_module)
with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.MEAN):
    # Turns off exploration
    td0 = env.rollout(max_steps=10, policy=exploration_policy)
with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.RANDOM):
    # Turns on exploration
    td0 = env.rollout(max_steps=10, policy=exploration_policy)


env = torchrl.envs.GymEnv("CartPole-v1", device=None)
dim_observation = env.observation_spec['observation'].shape[-1]
dim_action = env.action_spec.shape[-1]
tmp0 = torchrl.modules.MLP(dim_observation, dim_action, num_cells=[32, 32])
value_net = tensordict.nn.TensorDictModule(tmp0, in_keys=["observation"], out_keys=["action_value"])
action_net = torchrl.modules.QValueModule(action_space=env.action_spec)
policy = tensordict.nn.TensorDictSequential(
    value_net,
    torchrl.modules.QValueModule(action_space=env.action_spec),
    torchrl.modules.EGreedyModule(env.action_spec),
)
with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.RANDOM):
    td0 = env.rollout(max_steps=3, policy=policy)



# https://pytorch.org/rl/tutorials/getting-started-2.html
env = torchrl.envs.GymEnv("Pendulum-v1", device=None)
dim_observation = env.observation_spec["observation"].shape[-1]
dim_action = env.action_spec.shape[-1]
actor = torchrl.modules.Actor(torchrl.modules.MLP(dim_observation, dim_action, num_cells=[32, 32]))
tmp0 = torchrl.modules.MLP(dim_observation+dim_action, out_features=1, num_cells=[32, 32])
value_net = torchrl.modules.ValueOperator(tmp0, in_keys=["observation", "action"])
ddpg_loss = torchrl.objectives.DDPGLoss(actor_network=actor, value_network=value_net)

rollout = env.rollout(max_steps=100, policy=actor)
loss_vals = ddpg_loss(rollout)
# loss_vals: tensordict
#   loss_actor: (torch, float32, ())
#   loss_value: (torch, float32, ())
#   pred_value:
#   pred_value_max:
#   target_value
#   target_value_max
#   td_error

total_loss = sum(v for k, v in loss_vals.items() if k.startswith("loss_"))
optimizer = torch.optim.Adam(ddpg_loss.parameters())
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
updater = torchrl.objectives.SoftUpdate(ddpg_loss, eps=0.99)
updater.step()


## https://pytorch.org/rl/stable/tutorials/getting-started-3.html
env = torchrl.envs.GymEnv("CartPole-v1", device=None)
policy = torchrl.envs.utils.RandomPolicy(env.action_spec)
collector = torchrl.collectors.SyncDataCollector(env, policy, frames_per_batch=200, total_frames=-1)
collector.frames_per_batch #200
td0 = next(iter(collector))
td0['collector', 'traj_ids'] #[0,0,0,1,1,1,1,...]
# td0: tensordict
#   action: (torch, int64, (200,2))
#   collector: tensordict
#     traj_ids: (torch, int64, (200,))
#   done: (torch, bool, (200,1))
#   next: tensordict
#     done: (torch, bool, (200,1))
#     observation: (torch, float32, (200,4))
#     reward: (torch, float32, (200,1))
#     terminated: (torch, bool, (200,1))
#     truncated: (torch, bool, (200,1))
#   observation: (torch, float32, (200,4))
#   terminated: (torch, bool, (200,1))
#   truncated: (torch, bool, (200,1))

buffer = torchrl.data.replay_buffers.ReplayBuffer(storage=torchrl.data.replay_buffers.LazyMemmapStorage(max_size=1000))
# buffer.add(next(iter(collector))[0]) #single item
indices = buffer.extend(next(iter(collector))) #[0...199] [200...399]
td1 = buffer.sample(batch_size=30)


## https://pytorch.org/rl/stable/tutorials/getting-started-5.html
env = torchrl.envs.TransformedEnv(torchrl.envs.GymEnv("CartPole-v1", device=None), torchrl.envs.StepCounter())

value_mlp = torchrl.modules.MLP(out_features=env.action_spec.shape[-1], num_cells=[64, 64])
value_net = tensordict.nn.TensorDictModule(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = tensordict.nn.TensorDictSequential(value_net, torchrl.modules.QValueModule(spec=env.action_spec))
exploration_module = torchrl.modules.EGreedyModule(env.action_spec, annealing_num_steps=100_000, eps_init=0.5)
policy_explore = tensordict.nn.TensorDictSequential(policy, exploration_module)

init_rand_steps = 5120
collector = torchrl.collectors.SyncDataCollector(env, policy,
        frames_per_batch=1280, total_frames=-1, init_random_frames=init_rand_steps)
buffer = torchrl.data.ReplayBuffer(storage=torchrl.data.LazyTensorStorage(100_000))

loss = torchrl.objectives.DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = torch.optim.Adam(loss.parameters(), lr=0.02)
updater = torchrl.objectives.SoftUpdate(loss, eps=0.99)

total_count = 0
total_episodes = 0
for ind_batch, data in enumerate(collector):
    buffer.extend(data)
    max_length = data["next", "step_count"].max()
    if len(buffer) > init_rand_steps:
        for _ in range(20):
            sample = buffer.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            updater.step()
            exploration_module.step(sample.numel())
        total_count += data.numel()
        total_episodes += data["next", "done"].sum()
    print(f"[{ind_batch}] max_length={max_length}, len(buffer)={len(buffer)}")
    if ind_batch>50:
        break
print(f"solved after {total_count} steps, {total_episodes} episodes")


# mamba install -c conda-forge moviepy ffmpeg
logdir = next_tbd_dir(tag_create=False)
tmp0 = gym.make("CartPole-v1", render_mode="rgb_array")
hf0 = lambda x: True
tmp1 = gym.wrappers.RecordVideo(tmp0, video_folder=logdir, name_prefix="test", video_length=500, episode_trigger=hf0)
env_gym = gym.wrappers.RecordEpisodeStatistics(tmp1)
for ind0 in range(5):
    obs,info = env_gym.reset()
    ind_step = 0
    while True:
        td0 = tensordict.TensorDict({'observation':torch.tensor(obs, dtype=torch.float32)}, batch_size=[])
        action = policy(td0)['action'][1].item()
        obs, reward, terminated, truncated, info = env_gym.step(action)
        ind_step += 1
        if terminated or truncated:
            break
    print(f"Episode {ind0} finished after {ind_step} steps")
env_gym.close()
