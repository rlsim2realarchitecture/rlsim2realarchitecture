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

from meta_rl_tools.nets import PolNetLSTM, VNetLSTM
import meta_rl_tools.envs.pybullet_locomotors
from meta_rl_tools.envs.pybullet_locomotors.gym_locomotion_envs import AntBulletEnvRandomized
from meta_rl_tools.envs.wrappers import ObAcEnv, TimeLimitEnv
from common_args import common_args, ppo_args, constant_step_args, sequense_step_args, random_env_args, sim2real_args, change_batch_size_for_dp, create_env

parser = argparse.ArgumentParser()
parser = common_args(parser)
parser = ppo_args(parser)
parser = sequense_step_args(parser)
parser = random_env_args(parser)
parser = sim2real_args(parser)

parser.add_argument('--h_size', type=int, default=734)
parser.add_argument('--cell_size', type=int, default=70)

args = parser.parse_args()
args = change_batch_size_for_dp(args)
args.sequence = True

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

env, center_env = create_env(args)

ob_space = env.observation_space
ac_space = env.action_space

pol_net = PolNetLSTM(ob_space, ac_space, h_size=args.h_size, cell_size=args.cell_size)

logger.log(str(len(torch.nn.utils.parameters_to_vector(pol_net.parameters()))))

if isinstance(ac_space, gym.spaces.Box):
    pol = GaussianPol(ob_space, ac_space, pol_net, rnn=True, data_parallel=args.data_parallel, parallel_dim=1)
elif isinstance(ac_space, gym.spaces.Discrete):
    pol = CategoricalPol(ob_space, ac_space, pol_net, rnn=True, data_parallel=args.data_parallel, parallel_dim=1)
elif isinstance(ac_space, gym.spaces.MultiDiscrete):
    pol = MultiCategoricalPol(ob_space, ac_space, pol_net, rnn=True, data_parallel=args.data_parallel, parallel_dim=1)
else:
    raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')

if args.pol:
    pol.load_state_dict(torch.load(args.pol, map_location=lambda storage, loc: storage))

vf_net = VNetLSTM(ob_space, h_size=args.h_size, cell_size=args.cell_size)
vf = DeterministicSVfunc(ob_space, vf_net, rnn=True, data_parallel=args.data_parallel, parallel_dim=1)

if args.vf:
    vf.load_state_dict(torch.load(args.vf, map_location=lambda storage, loc: storage))

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)
if center_env is not None:
    center_sampler = EpiSampler(center_env, pol, num_parallel=1, seed=args.seed)

optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
if args.optim_pol:
    optim_pol.load_state_dict(torch.load(args.optim_pol, map_location=lambda storage, loc: storage))
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)
if args.optim_vf:
    optim_vf.load_state_dict(torch.load(args.optim_vf, map_location=lambda storage, loc: storage))

total_epi = 0
total_step = 0
max_mean_rew = -1e6
max_min_rew = -1e6
max_center_rew = -1e6
kl_beta = args.init_kl_beta
while args.max_epis > total_epi:
    with measure('sample'):
        epis = sampler.sample(pol, max_epis=args.max_epis_per_iter)
        if center_env is not None:
            center_epis = center_sampler.sample(pol, max_epis=5)
            center_rewards = [np.sum(epi['rews']) for epi in center_epis]
            logger.record_tabular_misc_stat('CenterReward', center_rewards)
            center_mean_rew = np.mean(center_rewards)
            if center_mean_rew > max_center_rew:
                torch.save(pol.state_dict(), os.path.join(
                    args.log, 'models', 'pol_max_center.pkl'))
                torch.save(vf.state_dict(), os.path.join(
                    args.log, 'models', 'vf_max_center.pkl'))
                torch.save(optim_pol.state_dict(), os.path.join(
                    args.log, 'models', 'optim_pol_max_center.pkl'))
                torch.save(optim_vf.state_dict(), os.path.join(
                    args.log, 'models', 'optim_vf_max_center.pkl'))
                max_center_rew = center_mean_rew

    rewards = [np.sum(epi['rews']) for epi in epis]

    mean_rew = np.mean(rewards)
    if mean_rew > max_mean_rew:
        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_max_mean.pkl'))
        torch.save(vf.state_dict(), os.path.join(
            args.log, 'models', 'vf_max_mean.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_max_mean.pkl'))
        torch.save(optim_vf.state_dict(), os.path.join(
            args.log, 'models', 'optim_vf_max_mean.pkl'))
        max_mean_rew = mean_rew

    min_rew = np.min(rewards)
    if min_rew > max_min_rew:
        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_max_min.pkl'))
        torch.save(vf.state_dict(), os.path.join(
            args.log, 'models', 'vf_max_min.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_max_min.pkl'))
        torch.save(optim_vf.state_dict(), os.path.join(
            args.log, 'models', 'optim_vf_max_min.pkl'))
        max_min_rew = min_rew

    torch.save(pol.state_dict(), os.path.join(
        args.log, 'models', 'pol_last.pkl'))
    torch.save(vf.state_dict(), os.path.join(
        args.log, 'models', 'vf_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(
        args.log, 'models', 'optim_pol_last.pkl'))
    torch.save(optim_vf.state_dict(), os.path.join(
        args.log, 'models', 'optim_vf_last.pkl'))
    if args.save_epis:
        import joblib
        joblib.dump(epis, os.path.join(args.log, 'last_epis.pkl'))

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
            if args.no_learning:
                result_dict = {}
            else:
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
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    del traj
del sampler
del center_sampler


