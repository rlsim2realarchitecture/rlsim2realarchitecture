import gym
import numpy as np


class ObAcEnv(gym.Env):
    def __init__(self, env, dim=0, max_steps=None):
        self.env = env
        if hasattr(env, 'original_env'):
            self.original_env = env.original_env
        else:
            self.original_env = env
        self.dim = dim
        ob_space = self.env.observation_space
        ac_space = self.env.action_space
        low = np.concatenate([ob_space.low, ac_space.low], axis=dim)
        high = np.concatenate([ob_space.high, ac_space.high], axis=dim)
        self.ob_space = gym.spaces.Box(low, high)
        self.ac_space = self.env.action_space
        self.current_index = 0
        self.max_steps = max_steps

    @property
    def observation_space(self):
        return self.ob_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        if hasattr(self.env, 'horizon'):
            return self.env._horizon

    def reset(self):
        ob = self.env.reset()
        self.current_index = 0
        zero_ac = np.zeros_like(self.action_space.low)
        ob = np.concatenate([ob, zero_ac], axis=self.dim)
        return ob

    def step(self, action):
        next_ob, reward, done, info = self.env.step(action)
        next_ob = np.concatenate([next_ob, action], axis=self.dim)
        self.current_index += 1
        if self.max_steps is not None:
            if self.current_index == self.max_steps:
                done = True
        return next_ob, reward, done, info

    def render(self):
        self.env.render()

    def terminate(self):
        self.env.terminate()
