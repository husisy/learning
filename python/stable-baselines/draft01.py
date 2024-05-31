import os
import numpy as np
import gymnasium as gym
import stable_baselines3
import stable_baselines3.common.results_plotter

from zzz233 import next_tbd_dir

## https://github.com/araffin/rl-tutorial-jnrr19 4-callbacks-hyperparameter-tuning
# from stable_baselines3 import A2C, SAC, PPO, TD3
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import BaseCallback

eval_env = gym.make("Pendulum-v1")
default_model = stable_baselines3.SAC("MlpPolicy", "Pendulum-v1", verbose=1, batch_size=64, policy_kwargs=dict(net_arch=[64, 64])).learn(8000)
mean_reward, std_reward = stable_baselines3.common.evaluation.evaluate_policy(default_model, eval_env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}") #-693.52 +/- 223.99

tuned_model = stable_baselines3.SAC("MlpPolicy", "Pendulum-v1", batch_size=256, verbose=1, policy_kwargs=dict(net_arch=[256, 256])).learn(8000)
mean_reward, std_reward = stable_baselines3.common.evaluation.evaluate_policy(tuned_model, eval_env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}") #-151.66 +/- 82.50

# class CustomCallback(stable_baselines3.common.callbacks.BaseCallback):
#     def __init__(self, verbose=0):
#         # verbose=0 (not output), verbose=1 (info), verbose=2 (debug)
#         super().__init__(verbose)
#         # self.model = None  # type: BaseRLModel
#         # self.training_env = None  #self.model.get_env(), type: Union[gym.Env, VecEnv, None]
#         # self.n_calls = 0  #Number of time the callback was called, type: int
#         # self.num_timesteps = 0  # type: int
#         # self.locals = None  #local varialbes, type: Dict[str, Any]
#         # self.globals = None  #global variables, type: Dict[str, Any]
#         # self.logger = None  #used to report things in the terminal, type: logger.Logger
#         # self.parent = None  #for event callback, type: Optional[BaseCallback]

#     def _on_training_start(self) -> None:
#         # called before the first rollout starts.
#         pass

#     def _on_rollout_start(self) -> None:
#         # triggered before collecting new samples
#         pass

#     def _on_step(self) -> bool:
#         # called by the model after each call to `env.step()`
#         # If the callback returns False, training is aborted early
#         return True

#     def _on_rollout_end(self) -> None:
#         # triggered before updating the policy
#         pass

#     def _on_training_end(self) -> None:
#         # triggered before exiting the `learn()` method
#         pass

class SimpleCallback(stable_baselines3.common.callbacks.BaseCallback):
    # a simple callback that can only be called twice
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._called = False

    def _on_step(self):
        if not self._called:
            print("callback - first call")
            self._called = True
            return True  # returns True, training continues.
        print("callback - second call")
        # returns False, training stops
        return False

model = stable_baselines3.SAC("MlpPolicy", "Pendulum-v1", verbose=1)
model.learn(8000, callback=SimpleCallback())



# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.results_plotter import load_results, ts2xy


# `stable_baselines3.common.callbacks.EvalCallback` is recommended in practice
class SaveOnBestTrainingRewardCallback(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self, check_freq:int, log_dir:str, verbose:int=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        if not os.path.exists(self.log_dir):
            pass

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            pd0 = stable_baselines3.common.results_plotter.load_results(self.log_dir) #pd.DataFrame
            x, y = stable_baselines3.common.results_plotter.ts2xy(pd0, "timesteps")
            if len(x) > 0:
                mean_reward = np.mean(y[-100:]) #last 100 episodes
                if self.verbose > 0:
                    print(f"num_timestep={self.num_timesteps}, best_mean_reward={self.best_mean_reward:.2f} - Last mean_reward={mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)
        return True


log_dir = next_tbd_dir()
env = stable_baselines3.common.env_util.make_vec_env("CartPole-v1", n_envs=1, monitor_dir=log_dir)
# it is equivalent to:
# env = gym.make('CartPole-v1')
# env = stable_baselines3.common.monitor.Monitor(env, log_dir)
# env = stable_baselines3.common.vec_env.DummyVecEnv([lambda: env])
callback = SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir, verbose=1)
model = stable_baselines3.A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=5000, callback=callback)
