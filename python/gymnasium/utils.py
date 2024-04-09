import os
import random
import collections
import numpy as np


def next_tbd_dir(dir0='tbd00', maximum_int=100000, tag_create:bool=True):
    if not os.path.exists(dir0):
        os.makedirs(dir0)
    tmp1 = [x for x in os.listdir(dir0) if x[:3]=='tbd']
    exist_set = {x[3:] for x in tmp1}
    while True:
        tmp1 = str(random.randint(1,maximum_int))
        if tmp1 not in exist_set:
            break
    tbd_dir = os.path.join(dir0, 'tbd'+tmp1)
    if tag_create:
        os.mkdir(tbd_dir)
    return tbd_dir


class QLearningAgent:
    def __init__(self, dim_action_space:int, learning_rate:float, initial_epsilon:float, epsilon_decay:float, final_epsilon:float, discount_factor:float=0.95, seed=None):
        self.np_rng = np.random.default_rng(seed)
        self.dim_action_space = int(dim_action_space)
        self.q_values = collections.defaultdict(lambda: np.zeros(dim_action_space))

        self.lr = learning_rate
        self.discount_factor = discount_factor #The discount factor for computing the Q-value

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs:tuple[int,int,bool]) -> int:
        # with probability epsilon return a random action to explore the environment
        if self.np_rng.uniform() < self.epsilon:
            # env.action_space.sample()
            ret = self.np_rng.integers(self.dim_action_space)
        else:
            ret = np.argmax(self.q_values[tuple(int(x) for x in obs)])
        return ret

    def update(self, obs:tuple[int,int,bool], action:int, reward:float, terminated:bool, next_obs:tuple[int,int,bool]):
        obs = tuple(int(x) for x in obs)
        next_obs = tuple(int(x) for x in next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        tmp0 = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * tmp0
        self.training_error.append(tmp0)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
