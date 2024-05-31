import numpy as np
import torch
import torchrl.modules
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensordict
import torchrl

plt.ion()

device = torch.device("cpu")


# from tensordict.nn import TensorDictModule
# from torchrl.objectives.utils import ValueEstimators
# from torchrl.objectives.utils import default_value_kwargs
# from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
# from torchrl.envs import CatTensors, DoubleToFloat, EnvCreator, InitTracker, ObservationNorm, ParallelEnv, RewardScaling, StepCounter, TransformedEnv
# from torchrl.envs.libs.dm_control import DMControlEnv
# from torchrl.envs.libs.gym import GymEnv
# from torchrl.data import CompositeSpec
# from torchrl.modules import ActorCriticWrapper, DdpgMlpActor, DdpgMlpQNet, OrnsteinUhlenbeckProcessWrapper, ProbabilisticActor, TanhDelta, ValueOperator
# from torchrl.objectives.utils import distance_loss
# from tensordict import TensorDict, TensorDictBase
# from torchrl.objectives import LossModule
# from torchrl.collectors import SyncDataCollector
# from torchrl.envs import ExplorationType


class DDPGLoss(torchrl.objectives.LossModule):

    default_value_estimator = torchrl.objectives.utils.ValueEstimators.TD0

    def __init__(self, actor_network:tensordict.nn.TensorDictModule, value_network:tensordict.nn.TensorDictModule) -> None:
        super().__init__()
        self.convert_to_functional(actor_network, "actor_network", create_target_params=True)
        self.convert_to_functional(value_network, "value_network",
                create_target_params=True, compare_against=list(actor_network.parameters()))
        self.actor_in_keys = actor_network.in_keys

        # Since the value we'll be using is based on the actor and value network,
        # we put them together in a single actor-critic container.
        actor_critic = torchrl.modules.ActorCriticWrapper(actor_network, value_network)
        self.actor_critic = actor_critic
        self.loss_function = "l2"

    def forward(self, input_tensordict: tensordict.TensorDictBase) -> tensordict.TensorDict:
        loss_value, td_error, pred_val, target_value = self.loss_value(input_tensordict)
        td_error = td_error.detach().unsqueeze(input_tensordict.ndimension())
        if input_tensordict.device is not None:
            td_error = td_error.to(input_tensordict.device)
        input_tensordict.set("td_error", td_error, inplace=True)
        loss_actor = self.loss_actor(input_tensordict)
        tmp0 = {
                "loss_actor": loss_actor.mean(),
                "loss_value": loss_value.mean(),
                "pred_value": pred_val.mean().detach(),
                "target_value": target_value.mean().detach(),
                "pred_value_max": pred_val.max().detach(),
                "target_value_max": target_value.max().detach(),
        }
        ret = tensordict.TensorDict(tmp0, batch_size=[])
        return ret

    def loss_value(self, td0):
        td_copy = td0.clone()

        # V(s, a)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)
        pred_val = td_copy.get("state_action_value").squeeze(-1)

        # we manually reconstruct the parameters of the actor-critic, where the first
        # set of parameters belongs to the actor and the second to the value function.
        tmp0 = {"module": {"0":self.target_actor_network_params, "1": self.target_value_network_params}}
        target_params = tensordict.TensorDict(tmp0,
            batch_size=self.target_actor_network_params.batch_size,
            device=self.target_actor_network_params.device)
        with target_params.to_module(self.value_estimator):
            target_value = self.value_estimator.value_estimate(td0).squeeze(-1)

        # Computes the value loss: L2, L1 or smooth L1 depending on `self.loss_function`
        ret = torchrl.objectives.utils.distance_loss(pred_val, target_value, loss_function=self.loss_function)
        td_error = (pred_val - target_value).pow(2)
        return ret, td_error, pred_val, target_value

    def loss_actor(self, td0) -> torch.Tensor:
        td_copy = td0.select(*self.actor_in_keys)
        # Get an action from the actor network: since we made it functional, we need to pass the params
        with self.actor_network_params.to_module(self.actor_network):
            td_copy = self.actor_network(td_copy)
        # get the value associated with that action
        with self.value_network_params.detach().to_module(self.value_network):
            td_copy = self.value_network(td_copy)
        return -td_copy.get("state_action_value")

    def make_value_estimator(self, value_type: torchrl.objectives.utils.ValueEstimators, **hyperparams):
        hp = dict(torchrl.objectives.utils.default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        value_key = "state_action_value"
        if value_type == torchrl.objectives.utils.ValueEstimators.TD1:
            self._value_estimator = torchrl.objectives.value.TD1Estimator(value_network=self.actor_critic, **hp)
        elif value_type == torchrl.objectives.utils.ValueEstimators.TD0:
            self._value_estimator = torchrl.objectives.value.TD0Estimator(value_network=self.actor_critic, **hp)
        elif value_type == torchrl.objectives.utils.ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == torchrl.objectives.utils.ValueEstimators.TDLambda:
            self._value_estimator = torchrl.objectives.value.TDLambdaEstimator(value_network=self.actor_critic, **hp)
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")
        self._value_estimator.set_keys(value=value_key)


env = torchrl.envs.GymEnv("HalfCheetah-v4")
# env = torchrl.envs.GymEnv("HalfCheetah-v4", from_pixels=True, pixels_only=True)
# env = torchrl.envs.GymEnv("HalfCheetah-v4", render_mode="human")
env = torchrl.envs.GymEnv("HalfCheetah-v4", render_mode="rgb_array")
# np0 = env.render() #(np,uint8,(480,480,3))
# fig,ax = plt.subplots()
# ax.imshow(np0)

make_env = lambda: torchrl.envs.GymEnv("HalfCheetah-v4", render_mode="rgb_array")



def make_transformed_env(env):
    """Apply transforms to the ``env`` (such as reward scaling and state normalization)."""
    env = torchrl.envs.TransformedEnv(env)

    # we append transforms one by one, although we might as well create the
    # transformed environment using the `env = TransformedEnv(base_env, transforms)` syntax.
    env.append_transform(torchrl.envs.RewardScaling(loc=0.0, scale=reward_scaling))
    out_key = "observation_vector"
    env.append_transform(torchrl.envs.CatTensors(in_keys=list(env.observation_spec.keys()), out_key=out_key))
    # we normalize the states, but for now let's just instantiate a stateless version of the transform
    env.append_transform(torchrl.envs.ObservationNorm(in_keys=[out_key], standard_normal=True))
    env.append_transform(torchrl.envs.DoubleToFloat())
    env.append_transform(torchrl.envs.StepCounter(max_frames_per_traj))
    # We need a marker for the start of trajectories for our Ornstein-Uhlenbeck (OU) exploration:
    env.append_transform(torchrl.envs.InitTracker())
    return env


reward_scaling = 5.0
max_frames_per_traj = 500
init_env_steps = 5000

# env = torchrl.envs.ParallelEnv(lambda: torchrl.envs.TransformedEnv(torchrl.envs.GymEnv("HalfCheetah-v4"), transforms), num_workers=4)
# env = torchrl.envs.TransformedEnv(torchrl.envs.ParallelEnv(lambda: torchrl.envs.GymEnv("HalfCheetah-v4"), num_workers=4), transforms)


def parallel_env_constructor(env_per_collector, transform_state_dict):
    if env_per_collector == 1:
        def make_t_env():
            env = make_transformed_env(make_env())
            env.transform[2].init_stats(3)
            env.transform[2].loc.copy_(transform_state_dict["loc"])
            env.transform[2].scale.copy_(transform_state_dict["scale"])
            return env

        env_creator = torchrl.envs.EnvCreator(make_t_env)
        return env_creator

    parallel_env = torchrl.envs.ParallelEnv(
        num_workers=env_per_collector,
        create_env_fn=torchrl.envs.EnvCreator(lambda: make_env()),
        create_env_kwargs=None,
        pin_memory=False,
    )
    env = make_transformed_env(parallel_env)
    # we call `init_stats` for a limited number of steps, just to instantiate the lazy buffers.
    env.transform[2].init_stats(3, cat_dim=1, reduce_dim=[0, 1])
    env.transform[2].load_state_dict(transform_state_dict)
    return env


def get_env_stats():
    """Gets the stats of an environment."""
    proof_env = make_transformed_env(make_env())
    t = proof_env.transform[2]
    t.init_stats(init_env_steps)
    transform_state_dict = t.state_dict()
    proof_env.close()
    return transform_state_dict


transform_state_dict = get_env_stats()
env_per_collector = 4

parallel_env = parallel_env_constructor(env_per_collector=env_per_collector, transform_state_dict=transform_state_dict)

def make_ddpg_actor(transform_state_dict, device="cpu"):
    proof_environment = make_transformed_env(make_env())
    proof_environment.transform[2].init_stats(3)
    proof_environment.transform[2].load_state_dict(transform_state_dict)

    actor_net = torchrl.modules.DdpgMlpActor(action_dim=proof_environment.action_spec.shape[-1])
    actor = tensordict.nn.TensorDictModule(actor_net, in_keys=["observation_vector"], out_keys=["param"])
    actor = torchrl.modules.ProbabilisticActor(actor, distribution_class=torchrl.modules.TanhDelta, in_keys=["param"],
            spec=torchrl.data.CompositeSpec(action=proof_environment.action_spec),
    ).to(device)

    # initialize lazy modules
    qnet = torchrl.modules.ValueOperator(in_keys=["observation_vector","action"], module=torchrl.modules.DdpgMlpQNet()).to(device)
    qnet(actor(proof_environment.reset().to(device)))
    return actor, qnet


actor, qnet = make_ddpg_actor(transform_state_dict=transform_state_dict, device=device)
actor_model_explore = torchrl.modules.OrnsteinUhlenbeckProcessWrapper(actor, annealing_num_steps=1_000_000).to(device)
if device == torch.device("cpu"):
    actor_model_explore.share_memory()


total_frames = 10_000  # 1_000_000
traj_len = 200
frames_per_batch = env_per_collector * traj_len
init_random_frames = 5000
num_collectors = 2


collector = torchrl.collectors.SyncDataCollector(parallel_env, policy=actor_model_explore, total_frames=total_frames,
    frames_per_batch=frames_per_batch, init_random_frames=init_random_frames, reset_at_each_iter=False,
    split_trajs=False, device=device, exploration_type=torchrl.envs.ExplorationType.RANDOM,
)

# from torchrl.trainers import Recorder


def make_recorder(actor_model_explore, transform_state_dict, record_interval):
    base_env = make_env()
    environment = make_transformed_env(base_env)
    environment.transform[2].init_stats(3)  # must be instantiated to load the state dict
    environment.transform[2].load_state_dict(transform_state_dict)

    recorder_obj = torchrl.trainers.Recorder(
        record_frames=1000,
        policy_exploration=actor_model_explore,
        environment=environment,
        exploration_type=torchrl.envs.ExplorationType.MEAN,
        record_interval=record_interval,
    )
    return recorder_obj


record_interval = 10

recorder = make_recorder(actor_model_explore, transform_state_dict, record_interval=record_interval)

# from torchrl.data.replay_buffers import LazyMemmapStorage, PrioritizedSampler, RandomSampler, TensorDictReplayBuffer
# from torchrl.envs import RandomCropTensorDict


def make_replay_buffer(buffer_size, batch_size, random_crop_len, prefetch=3, prb=False):
    if prb:
        sampler = torchrl.data.replay_buffers.PrioritizedSampler(max_capacity=buffer_size, alpha=0.7, beta=0.5)
    else:
        sampler = torchrl.data.replay_buffers.RandomSampler()
    tmp0 = torchrl.data.replay_buffers.LazyMemmapStorage(buffer_size, scratch_dir=buffer_scratch_dir)
    replay_buffer = torchrl.data.replay_buffers.TensorDictReplayBuffer(storage=tmp0,
            batch_size=batch_size, sampler=sampler, pin_memory=False, prefetch=prefetch,
            transform=torchrl.envs.RandomCropTensorDict(random_crop_len, sample_dim=1),
    )
    return replay_buffer


import tempfile

tmpdir = tempfile.TemporaryDirectory()
buffer_scratch_dir = tmpdir.name


def ceil_div(x, y):
    return -x // (-y)

buffer_size = 1_000_000
buffer_size = ceil_div(buffer_size, traj_len)

prb = False
update_to_data = 64
random_crop_len = 25

batch_size = ceil_div(64 * frames_per_batch, update_to_data * random_crop_len)

replay_buffer = make_replay_buffer(buffer_size=buffer_size, batch_size=batch_size,
        random_crop_len=random_crop_len, prefetch=3, prb=prb)

gamma = 0.99
lmbda = 0.9
tau = 0.001  # Decay factor for the target network

loss_module = DDPGLoss(actor, qnet)

loss_module.make_value_estimator(torchrl.objectives.utils.ValueEstimators.TDLambda, gamma=gamma, lmbda=lmbda)
