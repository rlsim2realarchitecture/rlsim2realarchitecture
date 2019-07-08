import os

import torch

from machina.envs import GymEnv, C2DEnv

from meta_rl_tools.envs.pybullet_locomotors.gym_locomotion_envs import HopperBulletEnvRandomized
from meta_rl_tools.envs.wrappers import ObAcEnv, MultiTimeStepEnv, TimeLimitEnv

from meta_rl_tools.envs.sim2real_envs.torque_pendulum import TorquePendulum
from meta_rl_tools.envs.sim2real_envs.torque_pendulum2 import TorquePendulum2

def centerize(l):
    return [(l[0] + l[1]) / 2, (l[0] + l[1]) / 2]

def common_args(parser):
    parser.add_argument('--log', type=str, default='garbage')
    parser.add_argument('--env_name', type=str, default='hopper_random')
    parser.add_argument('--no_c2d', action='store_true', default=True)
    parser.add_argument('--sequence', action='store_true', default=False)
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=256)
    parser.add_argument('--max_epis', type=int, default=10000000)
    parser.add_argument('--num_parallel', type=int, default=4)
    parser.add_argument('--data_parallel', action='store_true', default=False)
    parser.add_argument('--no_learning', action='store_true', default=False)
    parser.add_argument('--save_epis', action='store_true', default=False)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--pol', type=str, default='')
    parser.add_argument('--vf', type=str, default='')
    parser.add_argument('--optim_pol', type=str, default='')
    parser.add_argument('--optim_vf', type=str, default='')

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    return parser

def ppo_args(parser):
    parser.add_argument('--epoch_per_iter', type=int, default=10)
    parser.add_argument('--pol_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=3e-4)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--ppo_type', type=str,
                        choices=['clip', 'kl'], default='clip')
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--kl_targ', type=float, default=0.01)
    parser.add_argument('--init_kl_beta', type=float, default=1)
    return parser

def constant_step_args(parser):
    parser.add_argument('--timestep', type=int, default=64)
    parser.add_argument('--max_steps_per_iter', type=int, default=65536)
    parser.add_argument('--batch_size', type=int, default=8192)
    return parser

def sequense_step_args(parser):
    parser.add_argument('--max_epis_per_iter', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser

def random_env_args(parser):
    parser.add_argument('--no_all_random', action='store_true', default=False)
    parser.add_argument('--leg_range', type=float, nargs=2, default=[5, 20])
    parser.add_argument('--lateral_friction', type=float, nargs=2, default=[0.5, 1.5])
    parser.add_argument('--torque_power', type=float, nargs=2, default=[0.1, 3.0])
    parser.add_argument('--observation_noise_std', type=float, default=0.01)
    parser.add_argument('--easy_hard', type=str, default=None)
    return parser

def sim2real_args(parser):
    parser.add_argument('--goal_deg', type=float, default=20)
    parser.add_argument('--joint_damping_range', type=float, nargs=2, default=[0, 1])
    parser.add_argument('--linear_damping_range', type=float, nargs=2, default=[0, 10])
    parser.add_argument('--mass_range', type=float, nargs=2, default=[1, 20])
    parser.add_argument('--joint_friction_range', type=float, nargs=2, default=[0, 5])
    parser.add_argument('--lateral_friction_range', type=float, nargs=2, default=[0.2, 0.6])
    parser.add_argument('--push_force_range', type=float, nargs=2, default=[20000, 40000])
    parser.add_argument('--f_candidates', type=int, nargs='*', default=[25])
    parser.add_argument('--latency_range', type=int, nargs=2, default=[1, 4])
    parser.add_argument('--max_force_range', type=float, nargs=2, default=[30, 300])
    parser.add_argument('--use_wall', action='store_true', default=False)
    parser.add_argument('--max_seconds', type=int, default=20)
    parser.add_argument('--rew_scale', type=int, default=0.001)
    parser.add_argument('--speed_coef', type=float, default=0.1)
    parser.add_argument('--noise_scale', type=float, default=0.01)
    return parser


def change_batch_size_for_dp(args):
    if args.data_parallel and torch.cuda.is_available():
        num_device = torch.cuda.device_count()
        args.batch_size = args.batch_size // num_device * num_device
    return args

def create_env(args):
    if args.env_name == 'hopper_random':
        if args.easy_hard is not None:
            if args.easy_hard == 'easy':
                args.leg_range = [3.5, 4]
                args.torque_power = [0.6, 0.8]
            elif args.easy_hard == 'hard':
                args.leg_range = [1, 8]
                args.torque_power = [0.5, 1]
            elif args.easy_hard == 'center':
                args.leg_range = [1, 8]
                args.torque_power = [0.5, 1]
                args.leg_range = centerize(args.leg_range)
                args.lateral_friction = centerize(args.lateral_friction)
                args.torque_power = centerize(args.torque_power)
                args.observation_noise_std = 0

        env = HopperBulletEnvRandomized(
            leg_range=args.leg_range,
            lateral_friction=args.lateral_friction,
            torque_power=args.torque_power,
            obs_noise=args.observation_noise_std,
            all_random=not args.no_all_random,
            render=args.render)
        env = TimeLimitEnv(env)
        env = GymEnv(env, log_dir=os.path.join(
            args.log, 'movie'), record_video=args.record)
        center_env = HopperBulletEnvRandomized(
            leg_range=((args.leg_range[0] + args.leg_range[1]) / 2, (args.leg_range[0] + args.leg_range[1]) / 2),
            lateral_friction=((args.lateral_friction[0] + args.lateral_friction[1]) / 2, (args.lateral_friction[0] + args.lateral_friction[1]) / 2),
            torque_power=((args.torque_power[0] + args.torque_power[1]) / 2, (args.torque_power[0] + args.torque_power[1]) / 2),
            obs_noise=args.observation_noise_std,
            all_random=not args.no_all_random,
            render=args.render)
        center_env = TimeLimitEnv(center_env)
        center_env = GymEnv(center_env, log_dir=os.path.join(
            args.log, 'movie_center'), record_video=args.record)
    elif args.env_name == 'torque_pendulum':
        env = TorquePendulum(
            joint_damping_range=args.joint_damping_range, linear_damping_range=args.linear_damping_range,
            mass_range=args.mass_range, joint_friction_range=args.joint_friction_range,
            lateral_friction_range=args.lateral_friction_range, push_force_range=args.push_force_range,
            f_candidates=args.f_candidates, latency_range=args.latency_range,
            max_force_range=args.max_force_range,
            use_wall=args.use_wall,
            max_seconds=args.max_seconds, rew_scale=args.rew_scale, speed_coef=args.speed_coef, noise_scale=args.noise_scale,
            render=args.render,
        )
        env = GymEnv(env, log_dir=os.path.join(
            args.log, 'movie'), record_video=args.record)
    elif args.env_name == 'torque_pendulum2':
        if args.easy_hard is not None:
            if args.easy_hard == 'easy':
                args.mass_range = (0.05, 0.5)
                args.max_force_range = (2, 10)
        env = TorquePendulum2(
            goal_deg=args.goal_deg,
            mass_range=args.mass_range, f_candidates=args.f_candidates, latency_range=args.latency_range, max_force_range=args.max_force_range,
            max_seconds=args.max_seconds, noise_scale=args.noise_scale, render=args.render
        )
        env = GymEnv(env, log_dir=os.path.join(
            args.log, 'movie'), record_video=args.record)

    env.original_env.seed(args.seed)
    if 'center_env' in locals():
        center_env.original_env.seed(args.seed)

    env = ObAcEnv(env)
    if 'center_env' in locals():
        center_env = ObAcEnv(center_env)

    if not args.sequence:
        env = MultiTimeStepEnv(env, args.timestep)
        if 'center_env' in locals():
            center_env = MultiTimeStepEnv(center_env, args.timestep)

    if not args.no_c2d:
        env = C2DEnv(env)
        if 'center_env' in locals():
            center_env = C2DEnv(center_env)

    if not 'center_env' in locals():
        center_env = None

    return env, center_env
