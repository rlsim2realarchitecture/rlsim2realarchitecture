import gym
from gym.envs.registration import registry, make, spec
from meta_rl_tools.envs.pybullet_locomotors.gym_locomotion_envs import AntBulletEnvRandomized
def register(id,*args,**kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id,*args,**kvargs)

# ------------bullet-------------

