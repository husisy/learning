# https://pytorch.org/rl/tutorials/getting-started-0.html
import numpy as np
import torch
import torchrl
import tensordict

env = torchrl.envs.GymEnv("Pendulum-v1")

reset = env.reset()
reset_with_action = env.rand_action(reset)
reset_with_action["action"]
stepped_data = env.step(reset_with_action)


data = torchrl.envs.step_mdp(stepped_data)

rollout = env.rollout(max_steps=10)
transition = rollout[3]


transformed_env = torchrl.envs.TransformedEnv(env, torchrl.envs.StepCounter(max_steps=10))
rollout = transformed_env.rollout(max_steps=100)
rollout['step_count']
rollout["next", "truncated"]


# https://pytorch.org/rl/tutorials/getting-started-1.html
env = torchrl.envs.GymEnv("Pendulum-v1")
module = torch.nn.Linear(in_features=env.observation_spec['observation'].shape[-1], out_features=env.action_spec.shape[-1])
# module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
policy = tensordict.nn.TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)
rollout = env.rollout(max_steps=10, policy=policy)


env = torchrl.envs.GymEnv("Pendulum-v1")
module = torch.nn.Linear(in_features=env.observation_spec['observation'].shape[-1], out_features=env.action_spec.shape[-1])
policy = torchrl.modules.Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)


env = torchrl.envs.GymEnv("Pendulum-v1")
module = torchrl.modules.MLP(
    out_features=env.action_spec.shape[-1],
    num_cells=[32, 64],
    activation_class=torch.nn.Tanh,
)
policy = torchrl.modules.Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)

# probabilistic policy
env = torchrl.envs.GymEnv("Pendulum-v1")
backbone = torchrl.modules.MLP(in_features=3, out_features=2)
extractor = tensordict.nn.distributions.NormalParamExtractor()
module = torch.nn.Sequential(backbone, extractor)
td_module = tensordict.nn.TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
policy = torchrl.modules.ProbabilisticActor(
    td_module,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=torch.distributions.Normal,
    return_log_prob=True,
)

rollout = env.rollout(max_steps=10, policy=policy)

with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.MEAN):
    # takes the mean as action
    rollout = env.rollout(max_steps=10, policy=policy)
with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.RANDOM):
    # Samples actions according to the dist
    rollout = env.rollout(max_steps=10, policy=policy)



policy = torchrl.modules.Actor(torchrl.modules.MLP(3, 1, num_cells=[32, 64]))
exploration_module = torchrl.modules.EGreedyModule(spec=env.action_spec, annealing_num_steps=1000, eps_init=0.5)
# eps=1 (random) -> eps=0 (deterministic)
exploration_policy = tensordict.nn.TensorDictSequential(policy, exploration_module)

with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.MEAN):
    # Turns off exploration
    rollout = env.rollout(max_steps=10, policy=exploration_policy)
with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.RANDOM):
    # Turns on exploration
    rollout = env.rollout(max_steps=10, policy=exploration_policy)


env = torchrl.envs.GymEnv("CartPole-v1")
num_actions = env.action_spec.shape[-1]
tmp0 = torchrl.modules.MLP(out_features=num_actions, num_cells=[32, 32])
value_net = tensordict.nn.TensorDictModule(tmp0, in_keys=["observation"], out_keys=["action_value"])
# writes action values in our tensordict
action_net = torchrl.modules.QValueModule(action_space=env.action_spec)
# Reads the "action_value" entry by default
policy = tensordict.nn.TensorDictSequential(value_net, action_net)
rollout = env.rollout(max_steps=3, policy=policy)

policy_explore = tensordict.nn.TensorDictSequential(policy, torchrl.modules.EGreedyModule(env.action_spec))
with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.RANDOM):
    rollout_explore = env.rollout(max_steps=3, policy=policy_explore)
