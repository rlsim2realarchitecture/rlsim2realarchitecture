import gym
import numpy as np


class MultiTimeStepEnv(gym.Env):
    def __init__(self, env, timestep=32):
        self.env = env
        if hasattr(env, 'original_env'):
            self.original_env = env.original_env
        else:
            self.original_env = env
        self.timestep = timestep
        ob_space = self.env.observation_space
        low = np.concatenate([ob_space.low.reshape(1, -1) for _ in range(timestep)], axis=0)
        high = np.concatenate([ob_space.high.reshape(1, -1) for _ in range(timestep)], axis=0)
        self.ob_space = gym.spaces.Box(low, high)
        self.ac_space = self.env.action_space

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
        one_ob = self.env.reset()
        self.ob = np.array([one_ob] * self.timestep)
        return self.ob

    def step(self, action):
        next_one_ob, reward, done, info = self.env.step(action)
        ob = [next_one_ob]
        for i in range(self.timestep - 1):
            ob += [self.ob[i]]
        ob = np.array(ob)
        self.ob = ob
        return self.ob, reward, done, info

    def render(self):
        self.env.render()

    def terminate(self):
        self.env.terminate()

