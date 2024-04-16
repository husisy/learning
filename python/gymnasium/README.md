# gymnasium

1. link
   * [github](https://github.com/Farama-Foundation/Gymnasium)
   * [documentation](https://gymnasium.farama.org/)
2. install
   * `pip install gymnasium[box2d]==1.0.0a1`
   * `pip install gymnasium[atari]`
   * `pip install gymnasium[all]` fail on windows
   * `pip install moviepy glfw swig mujoco`
3. concept
   * markov decision process (MDP)
   * epsilon-greedy
   * Q-learning
   * episode
   * Deep Q Learning
   * [doi-link](https://doi.org/10.1007/BF00992696) Simple statistical gradient-following algorithms for connectionist reinforcement learning
   * `truncated`: The episode duration reaches max number of timesteps
   * `terminated`: Any of the state space values is no longer finite

others

[github/Practical_RL](https://github.com/yandexdataschool/Practical_RL)

[github/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning)

[github/cleanrl](https://github.com/vwxyzjn/cleanrl)

[github/Reinforcement-Learning](https://github.com/andri27-ts/Reinforcement-Learning)

[github/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)

## AgileRL

1. link
   * [github](https://github.com/AgileRL/AgileRL)

misc

[github/wandb](https://docs.wandb.ai/guides)

## torchrl

1. link
   * [documentation](https://pytorch.org/rl/)
   * [github](https://github.com/pytorch/rl)
   * [arxiv-link](https://arxiv.org/abs/2306.00577) TorchRL: A data-driven decision-making library for PyTorch
   * [github/torchdict](https://github.com/pytorch/tensordict)
2. concept
   * policy, model
   * TED: TorchRL Episode Data format
   * MDP: Markov Decision Process
   * PPO: Proximal Policy Optimization
3. simulation backend
   * `gymnaisum`
   * `google/dm_control`
   * `google/brax`

## google/brax

Massively parallel rigidbody physics simulation on accelerator hardware

1. link
   * [github](https://github.com/google/brax)

## deepmind/dm-control

1. link
   * [github](https://github.com/google-deepmind/dm_control)
