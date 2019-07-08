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

from meta_rl_tools.nets import PolNetLSTM


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

Hz = 25

parser = argparse.ArgumentParser()
parser.add_argument('--log', default='garbage')
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--pol', type=str, default='pol_lstm.pkl')
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--max_time', type=float, default=20)
parser.add_argument('--goal_deg', type=float, default=20)

parser.add_argument('---h_size', type=int, default=734)
parser.add_argument('--cell_size', type=int, default=70)
args = parser.parse_args()

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

r = redis.StrictRedis()

high = np.inf * np.ones(7)
observation_space = Box(low=-high, high=high, dtype=np.float32)
high = np.ones(1)
action_space = Box(low=-high, high=high)

pol_net = PolNetLSTM(observation_space, action_space, h_size=args.h_size, cell_size=args.cell_size)
pol = GaussianPol(observation_space, action_space, pol_net, data_parallel=args.data_parallel, rnn=True)

if args.pol:
    pol.load_state_dict(torch.load(args.pol, map_location=lambda storage, loc: storage))
else:
    raise Exception

pol.to(device)

pol.dp_run = False

pol.reset()

r.set('start', 'false')
while True:
    if r.get('start').decode('utf-8') == 'true':
        break

class Process(object):
    def run(self):
        joint_motor = float(r.get('joint_motor'))
        joint_encoder = float(r.get('joint_encoder'))

        joint_motor_vel = 0
        joint_encoder_vel = 0
        self.action_input = 0
        first_ob = [np.cos(joint_motor), np.sin(joint_motor), joint_motor_vel,
                    np.cos(joint_encoder), np.sin(joint_encoder), joint_encoder_vel,
                    self.action_input]
        self.ob = first_ob

        self.prev_joint_motor = joint_motor
        self.prev_joint_encoder = joint_encoder

        ac_real, _, _ = pol(torch.tensor(first_ob, dtype=torch.float, device=device))
        action = ac_real[0]

        r.set('torque', str(action.item()))
        self.action_input = action
        self.stop_event = threading.Event()

        epi = []

        self.all_time = 0
        self.start_time = time.time()

        self.done = False

        def callback():
            start_time = time.time()

            joint_motor = float(r.get('joint_motor'))
            joint_encoder = float(r.get('joint_encoder'))
            joint_motor_vel = (joint_motor - self.prev_joint_motor) * Hz
            joint_encoder_vel = (joint_encoder - self.prev_joint_encoder) * Hz
            self.prev_joint_motor = joint_motor
            self.prev_joint_encoder = joint_encoder

            self.done = np.abs(angle_normalize(joint_encoder - np.pi)) < np.abs(np.deg2rad(args.goal_deg))

            one_ob = [np.cos(joint_motor), np.sin(joint_motor), joint_motor_vel,
                        np.cos(joint_encoder), np.sin(joint_encoder), joint_encoder_vel,
                        self.action_input]
            self.ob = np.array(one_ob, dtype=np.float32)

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
            if self.all_time > args.max_time or self.done:
                break
            time.sleep(0.1)

        self.stop_event.set()
        time.sleep(1)
        r.set('torque', str(0))
        Path(args.log).mkdir(exist_ok=True)
        joblib.dump(epi, str(Path(args.log) / 'epi.pkl'))

p = Process()
p.run()


