import argparse
import threading
import time

from gym.spaces import MultiDiscrete, Box
import joblib
import numpy as np
from pathlib import Path
import pickle
import redis
import torch

from machina.utils import set_device
from machina.pols import MultiCategoricalPol, GaussianPol

from meta_rl_tools.nets import PolNetSNAILConstant

Hz = 25

parser = argparse.ArgumentParser()
parser.add_argument('--log', default='garbage')
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--pol', type=str, default='pol_snail.pkl')
parser.add_argument('--data_parallel', action='store_true', default=True)
parser.add_argument('--max_time', type=float, default=11)

parser.add_argument('--timestep', type=int, default=64)

parser.add_argument('--num_channels', type=int, default=110)
parser.add_argument('--num_keys', type=int, default=27)
parser.add_argument('--num_tc_fils', type=int, default=8)
parser.add_argument('--use_pe', action='store_true', default=False)
args = parser.parse_args()

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

r = redis.StrictRedis()

high = np.inf * np.ones(4)
observation_space = Box(low=-high, high=high, dtype=np.float32)
low = np.concatenate([observation_space.low.reshape(1, -1) for _ in range(args.timestep)], axis=0)
high = np.concatenate([observation_space.high.reshape(1, -1) for _ in range(args.timestep)], axis=0)
observation_space = Box(low, high)
high = np.ones(1)
action_space = Box(low=-high, high=high)

pol_net = PolNetSNAILConstant(observation_space, action_space, args.timestep, args.num_channels, num_keys=args.num_keys, num_tc_fils=args.num_tc_fils, use_pe=args.use_pe)
pol = GaussianPol(observation_space, action_space, pol_net, data_parallel=args.data_parallel)

if args.pol:
    pol.load_state_dict(torch.load(args.pol, map_location=lambda storage, loc: storage))
else:
    raise Exception

pol.to(device)

pol.dp_run = False

r.set('start', 'false')
while True:
    if r.get('start').decode('utf-8') == 'true':
        break

class Process(object):
    def run(self):
        joint_pendulum = float(r.get('joint_info'))

        joint_pendulum_vel = 0
        self.action_input = 0
        first_ob = [np.cos(joint_pendulum), np.sin(joint_pendulum), joint_pendulum_vel,
                    self.action_input]
        first_ob = np.array([first_ob] * args.timestep)
        self.ob = first_ob

        self.prev_joint_pendulum = joint_pendulum

        ac_real, _, _ = pol(torch.tensor(first_ob, dtype=torch.float, device=device))
        action = ac_real[0]

        r.set('torque', str(action.item()))
        self.action_input = action
        self.stop_event = threading.Event()

        epi = []

        self.all_time = 0
        self.start_time = time.time()

        def callback():
            start_time = time.time()

            joint_pendulum = float(r.get('joint_info'))
            joint_pendulum_vel = (joint_pendulum - self.prev_joint_pendulum) * Hz
            self.prev_joint_pendulum = joint_pendulum
            one_ob = [np.cos(joint_pendulum), np.sin(joint_pendulum), joint_pendulum_vel,
                      self.action_input]
            one_ob = np.array(one_ob)
            _ob = [one_ob]
            for i in range(args.timestep - 1):
                _ob += [self.ob[i]]
            self.ob = np.array(_ob, dtype=np.float32)

            ac_real, _, a_i = pol(torch.tensor(self.ob, dtype=torch.float, device=device))
            action = ac_real[0]

            r.set('torque', str(action.item()))
            self.action_input = action

            epi.append(dict(ob=self.ob))

            elapsed_time = time.time() - start_time
            intervel = max(1/Hz - elapsed_time, 0)
            if intervel == 0:
                print("inference time exceed {}sec".format(1/Hz))
            if not self.stop_event.is_set():
                threading.Timer(intervel, callback).start()

            self.all_time = time.time() - self.start_time

        callback()
        while True:
            if self.all_time > args.max_time:
                break
            time.sleep(0.1)

        self.stop_event.set()
        time.sleep(1)
        r.set('torque', str(0))
        Path(args.log).mkdir(exist_ok=True)
        joblib.dump(epi, str(Path(args.log) / 'epi.pkl'))

p = Process()
p.run()


