import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import pybullet_envs

import machina as mc
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.algos import ppo_clip, ppo_kl
from machina.vfuncs import DeterministicSVfunc
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device

from meta_rl_tools.nets import PolNetSNAIL, VNetSNAIL
import meta_rl_tools.envs.pybullet_locomotors
from meta_rl_tools.envs.wrappers import ObAcEnv


class CustomGaussianPol(GaussianPol):
    """
    Custom Policy Class with Gaussian distribution for SNAIL formats.
    """

    def __init__(self, ob_space, ac_space, net, normalize_ac=True, data_parallel=False, parallel_dim=0):
        rnn = True
        self.past_obs = None
        GaussianPol.__init__(self, ob_space, ac_space, net, rnn,
                         normalize_ac, data_parallel, parallel_dim)

    def forward(self, obs, hs=None, h_masks=None):
        obs = self._check_obs_shape(obs)
        
        time_seq, batch_size, *_ = obs.shape
        if hs is None:
            if self.hs is None:
                self.hs = self.net.init_hs(batch_size)
            if self.dp_run:
                self.hs = (self.hs[0].unsqueeze(0), self.hs[1].unsqueeze(0))
            hs = self.hs
        if h_masks is None:
            h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
        h_masks = h_masks.reshape(time_seq, batch_size, 1)
        
        if self.dp_run:
            mean, log_std, hs = self.dp_net(obs, hs, h_masks)
        else:
            mean, log_std, hs = self.net(obs, hs, h_masks)
        self.hs = hs
        
        log_std = log_std.expand_as(mean)
        ac = self.pd.sample(dict(mean=mean, log_std=log_std))
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(mean=mean, log_std=log_std, hs=hs)

    def deterministic_ac_real(self, obs, hs=None, h_masks=None):
        """
        action for deployment
        """
        obs = self._check_obs_shape(obs)

        time_seq, batch_size, *_ = obs.shape
        if hs is None:
            if self.hs is None:
                self.hs = self.net.init_hs(batch_size)
            hs = self.hs

        if h_masks is None:
            h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
        h_masks = h_masks.reshape(time_seq, batch_size, 1)

        mean, _, hs = self.net(obs, hs, h_masks)
        self.hs = hs

        mean_real = self.convert_ac_for_real(mean.detach().cpu().numpy())
        return mean_real, mean, dict(mean=mean, log_std=log_std, hs=hs)


class CustomDeterministicSVfunc(DeterministicSVfunc):
    """
    Custom Deterministic State Action Value Function Class for SNAIL formats.

    """

    def __init__(self, ob_space, net, data_parallel=False, parallel_dim=0):
        rnn = True
        super().__init__(ob_space, net, rnn, data_parallel, parallel_dim)

    def forward(self, obs, hs=None, h_masks=None):
        """
        Calculating values.
        """
        obs = self._check_obs_shape(obs)

        time_seq, batch_size, *_ = obs.shape
        if hs is None:
            if self.hs is None:
                self.hs = self.net.init_hs(batch_size)
            if self.dp_run:
                self.hs = (self.hs[0].unsqueeze(0), self.hs[1].unsqueeze(0))
            hs = self.hs

        if h_masks is None:
            h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
        h_masks = h_masks.reshape(time_seq, batch_size, 1)

        if self.dp_run:
            vs, hs = self.dp_net(obs, hs, h_masks)
        else:
            vs, hs = self.net(obs, hs, h_masks)
        self.hs = hs
        
        return vs.squeeze(-1), dict(mean=vs.squeeze(-1), hs=hs)


parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='AntBulletEnvCustom-v0')
parser.add_argument('--c2d', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=10000000)
parser.add_argument('--num_parallel', type=int, default=4)
parser.add_argument('--data_parallel', action='store_true', default=False)

parser.add_argument('--pol', type=str, default='')
parser.add_argument('--vf', type=str, default='')

parser.add_argument('--max_episodes_per_iter', type=int, default=128)
parser.add_argument('--num_epi_per_seq', type=int, default=2)
parser.add_argument('--max_steps_per_epi', type=int, default=200)
parser.add_argument('--epoch_per_iter', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--pol_lr', type=float, default=1e-4)
parser.add_argument('--vf_lr', type=float, default=3e-4)
parser.add_argument('--cuda', type=int, default=-1)

parser.add_argument('--max_grad_norm', type=float, default=10)

parser.add_argument('--ppo_type', type=str,
                    choices=['clip', 'kl'], default='clip')

parser.add_argument('--clip_param', type=float, default=0.2)

parser.add_argument('--kl_targ', type=float, default=0.01)
parser.add_argument('--init_kl_beta', type=float, default=1)

parser.add_argument('--gamma', type=float, default=0.995)
parser.add_argument('--lam', type=float, default=1)
args = parser.parse_args()

if not os.path.exists(args.log):
    os.mkdir(args.log)

with open(os.path.join(args.log, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

if not os.path.exists(os.path.join(args.log, 'models')):
    os.mkdir(os.path.join(args.log, 'models'))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

# env = GymEnv(args.env_name, log_dir=os.path.join(
#     args.log, 'movie'), record_video=args.record)
env = GymEnv(args.env_name)
env.env.seed(args.seed)
env = ObAcEnv(env, max_steps=args.max_steps_per_epi)
if args.c2d:
    env = C2DEnv(env)

ob_space = env.observation_space
ac_space = env.action_space

pol_net = PolNetSNAIL(ob_space, ac_space, T=args.max_steps_per_epi, n_episodes=args.num_epi_per_seq, device=device)

if isinstance(ac_space, gym.spaces.Box):
    pol = CustomGaussianPol(ob_space, ac_space, pol_net, data_parallel=args.data_parallel, parallel_dim=1)
else:
    raise ValueError('Only Box are supported')

'''
# TODO: Implementation
elif isinstance(ac_space, gym.spaces.Discrete):
    pol = CategoricalPol(ob_space, ac_space, pol_net, rnn=True, data_parallel=args.data_parallel, parallel_dim=1)
elif isinstance(ac_space, gym.spaces.MultiDiscrete):
    pol = MultiCategoricalPol(ob_space, ac_space, pol_net, rnn=True, data_parallel=args.data_parallel, parallel_dim=1)
else:
    raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')
'''

if args.pol:
    pol.load_state_dict(torch.load(args.pol, map_location=lambda storage, loc: storage))

vf_net = VNetSNAIL(ob_space, T=args.max_steps_per_epi, n_episodes=args.num_epi_per_seq, device=device)
vf = CustomDeterministicSVfunc(ob_space, vf_net, data_parallel=args.data_parallel, parallel_dim=1)

if args.vf:
    vf.load_state_dict(torch.load(args.vf, map_location=lambda storage, loc: storage))

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)

optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)

total_epi = 0
total_step = 0
max_rew = -1e6
kl_beta = args.init_kl_beta
while args.max_episodes > total_epi:
    with measure('sample'):
        epis = sampler.sample(pol, max_episodes=args.max_episodes_per_iter)
    with measure('train'):
        traj = Traj()
        traj.add_epis(epis)

        traj = ef.compute_vs(traj, vf)
        traj = ef.compute_rets(traj, args.gamma)
        traj = ef.compute_advs(traj, args.gamma, args.lam)
        traj = ef.centerize_advs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()
        
        if args.data_parallel:
            pol.dp_run = True
            vf.dp_run = True

        if args.ppo_type == 'clip':
            result_dict = ppo_clip.train(traj=traj, pol=pol, vf=vf, clip_param=args.clip_param,
                                         optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size, max_grad_norm=args.max_grad_norm)
        else:
            result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=kl_beta, kl_targ=args.kl_targ,
                                       optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size, max_grad_norm=args.max_grad_norm)
            kl_beta = result_dict['new_kl_beta']

        if args.data_parallel:
            pol.dp_run = False
            vf.dp_run = False

    total_epi += traj.num_epi
    step = traj.num_step
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_max.pkl'))
        torch.save(vf.state_dict(), os.path.join(
            args.log, 'models', 'vf_max.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_max.pkl'))
        torch.save(optim_vf.state_dict(), os.path.join(
            args.log, 'models', 'optim_vf_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(
        args.log, 'models', 'pol_last.pkl'))
    torch.save(vf.state_dict(), os.path.join(
        args.log, 'models', 'vf_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(
        args.log, 'models', 'optim_pol_last.pkl'))
    torch.save(optim_vf.state_dict(), os.path.join(
        args.log, 'models', 'optim_vf_last.pkl'))
    del traj
del sampler
