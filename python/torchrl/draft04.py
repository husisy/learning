# https://pytorch.org/rl/tutorials/pendulum.html
import collections
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

import torchrl
import tensordict

plt.ion()

def make_composite_from_td(td0):
    # custom function to convert a ``tensordict`` in a similar spec structure of unbounded values.
    tmp0 = {k:make_composite_from_td(t) if isinstance(t, tensordict.TensorDictBase) else
                 torchrl.data.UnboundedContinuousTensorSpec(dtype=t.dtype, device=t.device, shape=t.shape) for k,t in td0.items()}
    ret = torchrl.data.CompositeSpec(tmp0, shape=td0.shape)
    return ret

def torchrl_env_reset_wrapper(hf0):
    def _reset(self, tensordict):
        # "tensordict" is in the kwargs (cannot rename)
        return hf0(self, tensordict)
    return _reset

def torchrl_env_transform_reset_wrapper(hf0):
    def _reset(self, tensordict:tensordict.TensorDictBase, td_reset:tensordict.TensorDictBase) -> tensordict.TensorDictBase:
        # "tensordict" is in the kwargs (cannot rename)
        return hf0(self, tensordict, td_reset)
    return _reset

class PendulumEnv(torchrl.envs.EnvBase):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    batch_locked = False
    # not enforce the input `td0` to have the same batch size as the environment

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()
        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        self.set_seed(seed)

    @staticmethod
    def gen_params(g=10.0, batch_size=None) -> tensordict.TensorDictBase:
        tmp0 = {"max_speed": 8.0, "max_torque": 2.0, "dt": 0.05, "g": g, "m": 1.0, "l": 1.0}
        td = tensordict.TensorDict({"params": tensordict.TensorDict(tmp0, [])}, [])
        if batch_size: #bool(None) is False, bool([]) is False
            td = td.expand(batch_size).contiguous()
        return td

    def _make_spec(self, td_params):
        # self.output_spec["full_observation_spec"]
        tmp0 = torchrl.data.BoundedTensorSpec(low=-torch.pi, high=torch.pi, shape=(), dtype=torch.float32) #TODO this is bad
        tmp1 = torchrl.data.BoundedTensorSpec(low=-td_params["params", "max_speed"], high=td_params["params", "max_speed"], shape=(), dtype=torch.float32)
        self.observation_spec = torchrl.data.CompositeSpec(th=tmp0, thdot=tmp1, params=make_composite_from_td(td_params["params"]), shape=())
        self.state_spec = self.observation_spec.clone() #stateless environment
        # self.input_spec["full_action_spec"]
        self.action_spec = torchrl.data.BoundedTensorSpec(low=-td_params["params", "max_torque"], high=td_params["params", "max_torque"], shape=(1,), dtype=torch.float32) #TODO bad shape
        self.reward_spec = torchrl.data.UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))
        # self.done_spec

    @torchrl_env_reset_wrapper
    def _reset(self, td0):
        # non batch-locked environments: "td0.shape" dictates the number of simulators run simultaneously
        # other contexts: "self.batch_size" instead
        if td0 is None or td0.is_empty():
            td0 = self.gen_params(batch_size=self.batch_size)
        high_th = torch.tensor(np.pi, device=self.device)
        high_thdot = torch.tensor(1.0, device=self.device)
        low_th = -high_th
        low_thdot = -high_thdot

        th = (torch.rand(td0.shape, generator=self.rng, device=self.device) * (high_th - low_th) + low_th)
        thdot = (torch.rand(td0.shape, generator=self.rng, device=self.device) * (high_thdot - low_thdot) + low_thdot)
        ret = tensordict.TensorDict(dict(th=th, thdot=thdot, params=td0["params"]), batch_size=td0.shape)
        # if "done" is not present, it will be filled as False
        return ret

    # stateless environment. In stateful environments, the first argument would be "self"
    @staticmethod
    def _step(td0):
        th, thdot = td0["th"], td0["thdot"]
        g_force = td0["params", "g"]
        mass = td0["params", "m"]
        length = td0["params", "l"]
        dt = td0["params", "dt"]
        u = td0["action"].squeeze(-1).clamp(-td0["params", "max_torque"], td0["params", "max_torque"])
        tmp0 = ((th + torch.pi) % (2 * torch.pi)) - torch.pi #(-pi, pi]
        costs = tmp0**2 + 0.1 * thdot**2 + 0.001 * (u**2)

        tmp0 = thdot + (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length**2) * u) * dt
        new_thdot = tmp0.clamp(-td0["params", "max_speed"], td0["params", "max_speed"])
        new_th = th + new_thdot * dt
        reward = -costs.view(*td0.shape, 1)
        done = torch.zeros_like(reward, dtype=torch.bool)
        tmp0 = {"th":new_th, "thdot":new_thdot, "params":td0["params"], "reward":reward, "done":done}
        out = tensordict.TensorDict(tmp0, td0.shape)
        return out

    def _set_seed(self, seed:int|None=None):
        self.rng = torch.Generator() if (seed is None) else torch.manual_seed(seed)


env = PendulumEnv()
env.observation_spec
env.state_spec
env.reward_spec

td0 = env.reset()
# tensordict: batch_size=()
#   done: (torch, bool, (1))
#   params: batch_size=()
#     dt,g,l,m,max_speed,max_torque: (torch, float32, ())
#   terminated: (torch, bool, (1))
#   th: (torch, float32, ())
#   thdot: (torch, float32, ())
td0 = env.rand_step(td0) #in-place update
# tensordict: batch_size=()
#   action: (torch, float32, (1))
#   next: batch_size=()
#     done: (torch, bool, (1))
#     params (as above)
#     reward: (torch, float32, (1))
#     terminated: (torch, bool, (1))
#     th: (torch, float32, ())
#     thdot: (torch, float32, ())
td0['th'], td0['next','th'] #NOT same

# ``Unsqueeze`` the observations that we will concatenate
tmp0 = torchrl.envs.UnsqueezeTransform(unsqueeze_dim=-1, in_keys=["th", "thdot"], in_keys_inv=["th", "thdot"])
env = torchrl.envs.TransformedEnv(env, tmp0)
td0 = env.reset()
# tensordict: batch_size=()
#   done: (torch, bool, (1))
#   params: batch_size=()
#     dt,g,l,m,max_speed,max_torque: (torch, float32, ())
#   terminated: (torch, bool, (1))
#   th: (torch, float32, (1))
#   thdot: (torch, float32, (1))

class SinTransform(torchrl.envs.Transform):
    def _apply_transform(self, obs:torch.Tensor):
        return obs.sin()

    # The transform must also modify the data at reset time
    @torchrl_env_transform_reset_wrapper
    def _reset(self, td0, td_reset):
        return self._call(td_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @torchrl.envs.transforms.transforms._apply_to_composite
    def transform_observation_spec(self, observation_spec):
        ret = torchrl.data.BoundedTensorSpec(low=-1, high=1, shape=observation_spec.shape, dtype=observation_spec.dtype, device=observation_spec.device)
        return ret


class CosTransform(torchrl.envs.Transform):
    def _apply_transform(self, obs:torch.Tensor):
        return obs.cos()

    @torchrl_env_transform_reset_wrapper
    def _reset(self, td0, td_reset):
        return self._call(td_reset)

    @torchrl.envs.transforms.transforms._apply_to_composite
    def transform_observation_spec(self, observation_spec):
        ret = torchrl.data.BoundedTensorSpec(low=-1, high=1, shape=observation_spec.shape, dtype=observation_spec.dtype, device=observation_spec.device)
        return ret

env.append_transform(SinTransform(in_keys=["th"], out_keys=["sin"]))
env.append_transform(CosTransform(in_keys=["th"], out_keys=["cos"]))


cat_transform = torchrl.envs.CatTensors(in_keys=["sin", "cos", "thdot"], dim=-1, out_key="observation", del_keys=False) #sorted alphabetically
env.append_transform(cat_transform)
torchrl.envs.utils.check_env_specs(env)

rollout = []
td0 = env.reset()
for _ in range(100):
    td0["action"] = env.action_spec.rand()
    env.step(td0) #in-place
    rollout.append(td0.clone())
    td0 = torchrl.envs.utils.step_mdp(td0, keep_other=True) #out-of-place
rollout = torch.stack(rollout)


batch_size = 10
td0 = env.reset(env.gen_params(batch_size=batch_size))
td0 = env.rand_step(td0)
# auto_reset=False: we're executing the reset out of the ``rollout`` call
rollout = env.rollout(3, auto_reset=False, tensordict=env.reset(env.gen_params(batch_size=batch_size)))


n_observation_space = env.observation_spec['observation'].shape[0]
n_action_space = env.action_spec.shape[0]
net = torch.nn.Sequential(
    torch.nn.Linear(n_observation_space, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1),
)
policy = tensordict.nn.TensorDictModule(net, in_keys=["observation"], out_keys=["action"])


# fully differentiable simulator
batch_size = 32
num_total_frame = 20_000
optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_total_frame)
logs = collections.defaultdict(list)

with tqdm(range(num_total_frame // batch_size)) as pbar:
    for _ in pbar:
        optimizer.zero_grad()
        td0 = env.reset(env.gen_params(batch_size=[batch_size]))
        rollout = env.rollout(100, policy, tensordict=td0, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        pbar.set_description(f"reward: {traj_return: 4.4f}, last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}")
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[:, -1]["next", "reward"].mean().item())
        scheduler.step()

fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,5))
ax0.plot(logs["return"])
ax0.set_title("returns")
ax0.set_xlabel("iteration")
ax1.plot(logs["last_reward"])
ax1.set_title("last reward")
ax1.set_xlabel("iteration")
