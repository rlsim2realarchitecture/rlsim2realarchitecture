import time
import gym

class TimeLimitEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, env, max_episode_steps=1000):
        self.env = env
        if hasattr(env, 'original_env'):
            self.original_env = env.original_env
        else:
            self.original_env = env
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = 0

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        if hasattr(self.env, 'horizon'):
            return self.env._horizon

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            return True
        return False

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._past_limit():
            done = True 

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def terminate(self):
        self.env.terminate()

