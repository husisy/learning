
import numpy as np
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import tensordict
import torchrl

plt.ion()

frames_per_batch = 1000
total_frames = 100_000
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = 0.2 #clip value for PPO loss: see the equation in the intro for more context.
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

base_env = torchrl.envs.GymEnv("InvertedDoublePendulum-v4") #device=device
# normalize observations
tmp0 = torchrl.envs.Compose(
        torchrl.envs.ObservationNorm(in_keys=["observation"]),
        torchrl.envs.DoubleToFloat(),
        torchrl.envs.StepCounter(),
)
env = torchrl.envs.TransformedEnv(base_env, tmp0)
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
env.transform[0].loc.shape #11
env.observation_spec
env.reward_spec
env.input_spec
env.action_spec

torchrl.envs.ObservationNorm()

torchrl.envs.utils.check_env_specs(env)

rollout = env.rollout(3)
# rollout: tensordict
#   action: (torch, float32, (3,1))
#   done: (torch, bool, (3,1))
#   next:
#     done: (torch, bool, (3,1))
#     observation: (torch, float32, (3,11))
#     reward: (torch, float32, (3,1))
#     step_count: (torch, int64, (3,1))
#     terminated: (torch, bool, (3,1))
#     truncated: (torch, bool, (3,1))
#   observation: (torch, float32, (3,11))
#   step_count: (torch, int64, (3,1))
#   terminated: (torch, bool, (3,1))
#   truncated: (torch, bool, (3,1))

n_observation = env.observation_spec['observation'].shape[0]
n_action_space = env.action_spec.shape[-1]
actor_net = torch.nn.Sequential(
    torch.nn.Linear(n_observation, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 2*n_action_space),
    tensordict.nn.NormalParamExtractor(),
)
policy_module = torchrl.modules.ProbabilisticActor(
    module=tensordict.nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"]),
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=torchrl.modules.TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.low,
        "max": env.action_spec.space.high,
    },
    return_log_prob=True, # log-prob for the numerator of the importance weights
)


value_net = torch.nn.Sequential(
    torch.nn.Linear(n_observation, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 1),
)
value_module = torchrl.modules.ValueOperator(module=value_net, in_keys=["observation"])
policy_module(env.reset()) #tensordict
value_module(env.reset()) #tensordict

collector = torchrl.collectors.SyncDataCollector(env, policy_module,
        frames_per_batch=frames_per_batch, total_frames=total_frames, split_trajs=False)

replay_buffer = torchrl.data.ReplayBuffer(
    storage=torchrl.data.LazyTensorStorage(max_size=frames_per_batch),
    sampler=torchrl.data.SamplerWithoutReplacement(),
)

advantage_module = torchrl.objectives.value.GAE(gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True)
loss_module = torchrl.objectives.ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optimizer = torch.optim.Adam(loss_module.parameters(), 3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_frames // frames_per_batch, 0)
# optim.param_groups[0]["lr"]

logs = collections.defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""
for i, data_i in enumerate(collector):
    # data_i: tensordict
    #   action: (torch, float32, (1000,1))
    #   collector:
    #     traj_ids: (torch, int64, (1000,))
    #   done: (torch, bool, (1000,1))
    #   loc: (torch, float32, (1000,1))
    #   next:
    #     done: (torch, bool, (1000,1))
    #     observation: (torch, float32, (1000,11))
    #     reward: (torch, float32, (1000,1))
    #     step_count: (torch, int64, (1000,1))
    #     terminated: (torch, bool, (1000,1))
    #     truncated: (torch, bool, (1000,1))
    #   observation: (torch, float32, (1000,11))
    #   sample_log_prob: (torch, float32, (1000,))
    #   scale: (torch, float32, (1000,1))
    #   step_count: (torch, int64, (1000,1))
    #   terminated: (torch, bool, (1000,1))
    #   truncated: (torch, bool, (1000,1))

    for _ in range(num_epochs):
        # re-compute "advantage" signal at each epoch as its value depends on the value network which is updated in the inner loop.
        advantage_module(data_i) #in-place
        # advantage: (torch, float32, (1000,1))
        # next:
        #   state_value: (torch, float32, (1000,1))
        # value_target: (torch, float32, (1000,1))

        replay_buffer.extend(data_i.clone()) #whether necessay to clone?
        for _ in range(frames_per_batch // sub_batch_size):
            optimizer.zero_grad()
            tmp0 = loss_module(replay_buffer.sample(sub_batch_size))
            # tmp0: tensordict
            #   ESS: (torch, float32, ())
            #   entropy: (torch, float32, ())
            #   loss_critic: (torch, float32, ())
            #   loss_entropy: (torch, float32, ())
            #   loss_objective: (torch, float32, ())
            (tmp0["loss_objective"] + tmp0["loss_critic"] + tmp0["loss_entropy"]).backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
            optimizer.step()
    scheduler.step()

    if i % 10 == 0:
        # Evaluation: execute the policy without exploration (take the expected value of the action distribution) for a given number of steps
        with torchrl.envs.set_exploration_type(torchrl.envs.ExplorationType.MEAN), torch.no_grad():
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval_reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval_max_step"].append(eval_rollout["step_count"].max().item())
            eval_str = f"eval_reward={logs['eval_reward'][-1]:.3f}, eval_max_step={logs['eval_max_step'][-1]}"
    logs["reward"].append(data_i["next", "reward"].mean().item())
    logs["step_count"].append(data_i["step_count"].max().item())
    logs['lr'].append(optimizer.param_groups[0]["lr"])
    pbar.set_description(f"{eval_str}, reward={logs['reward'][-1]:.3f}, max_step={logs['step_count'][-1]}")
    pbar.update(data_i.numel())
pbar.close()

fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(9,4))
ax0.plot(logs["reward"], label='train')
tmp0 = np.linspace(0, len(logs["reward"])-1, len(logs["eval_reward"]))
ax0.plot(tmp0, logs["eval_reward"], label='eval')
ax0.set_title("rewards")
ax0.legend()
ax1.plot(logs["step_count"], label='train')
tmp0 = np.linspace(0, len(logs["step_count"])-1, len(logs["eval_max_step"]))
ax1.plot(tmp0, logs["eval_max_step"], label='eval') #should be around 1000 (maximum set by the environment)
ax1.set_title("max step count")
ax1.legend()
fig.tight_layout()
