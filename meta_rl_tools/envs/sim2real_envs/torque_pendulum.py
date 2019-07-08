from collections import deque
from pathlib import Path

from gym.core import Env
from gym.spaces import MultiDiscrete, Box
import numpy as np
import pybullet as p

from meta_rl_tools.envs.sim2real_envs.data import pendulum_from_template


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


class TorquePendulum(Env):
    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second': 20
        }

    def __init__(self, joint_damping_range=(0, 0.1), linear_damping_range=(0, 1), mass_range=(0.1, 10), joint_friction_range=(0, 5),
        lateral_friction_range=(0, 1.0), push_force_range=(10000, 30000), # randomized params for dynamics
        f_candidates=(20, ), latency_range=(1, 6), max_force_range=(50, 2000),# randomized params for control
        use_wall=False,
        max_seconds=20, rew_scale=0.001, speed_coef=0.01, noise_scale=0,
        render=False):

        self.joint_damping_range = joint_damping_range
        self.linear_damping_range = linear_damping_range
        self.mass_range = mass_range
        self.joint_friction_range = joint_friction_range
        self.lateral_friction_range = lateral_friction_range
        self.push_force_range = push_force_range

        self.f_candidates = f_candidates
        self.latency_range = latency_range
        self.max_force_range = max_force_range

        self.use_wall = use_wall

        self.max_seconds = max_seconds
        self.rew_scale = rew_scale
        self.speed_coef = speed_coef
        self.noise_scale = noise_scale

        if render:
            self.connect = p.GUI
        else:
            self.connect = p.DIRECT
        p.connect(self.connect)

        high = np.ones(1, dtype=np.float32)
        self.action_space = Box(low=-high, high=high, dtype=np.float32)
        high = np.inf * np.ones(3)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

        self.urdf_path = None

    def __del__(self):
        if self.urdf_path is not None:
            self.urdf_path.unlink()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.resetDebugVisualizerCamera(0.5, 90, 0, [0,0,0.2])

        self.joint_damping = np.random.uniform(*self.joint_damping_range)
        self.joint_friction = np.random.uniform(*self.joint_friction_range)
        if self.urdf_path is not None:
            self.urdf_path.unlink()
        self.urdf_path = urdf_path = pendulum_from_template(self.joint_damping, self.joint_friction)
        self.robot_id = p.loadURDF(str(urdf_path), [0, 0, 0], useFixedBase=True)
        # deactivate torque
        p.setJointMotorControl2(self.robot_id, 1, controlMode=p.VELOCITY_CONTROL, force=0)

        self.wall_id = p.loadURDF(str(Path(__file__).parent / 'data' / 'models' / 'friction_wall.urdf'), [-0.05-0.00, 0, 0.2], useFixedBase=True)

        init_joint_pendulum = np.random.uniform(-np.pi / 8, np.pi / 8)
        p.resetJointState(self.robot_id, 1, init_joint_pendulum)

        self.linear_damping = np.random.uniform(*self.linear_damping_range)
        self.mass = np.random.uniform(*self.mass_range)
        p.changeDynamics(self.robot_id, 1, mass=0.01, linearDamping=self.linear_damping)
        p.changeDynamics(self.robot_id, 2, mass=self.mass, linearDamping=self.linear_damping)

        self.lateral_friction = np.random.uniform(*self.lateral_friction_range)
        p.changeDynamics(self.robot_id, 1, lateralFriction=self.lateral_friction)

        self.push_force = np.random.uniform(*self.push_force_range)
        if self.use_wall:
            target_position = 0
        else:
            target_position = -1
        p.setJointMotorControl2(self.wall_id, 0, controlMode=p.POSITION_CONTROL, targetPosition=target_position, force=self.push_force)

        self.f = np.random.choice(self.f_candidates)
        self.num_step = int(240 / self.f)

        self.prev_joint_pendulum = joint_pendulum = p.getJointState(self.robot_id, 1)[0]
        joint_pendulum_vel = 0

        self.latency = np.random.randint(self.latency_range[0], self.latency_range[1])
        self.action_seq = deque(maxlen=self.latency + 1)
        for _ in range(self.latency + 1):
            self.action_seq.append(0)

        self.max_force = np.random.uniform(self.max_force_range[0], self.max_force_range[1])

        ob = np.array([np.cos(joint_pendulum), np.sin(joint_pendulum), joint_pendulum_vel,])
        ob += np.random.normal(0, self.noise_scale, ob.shape)

        for _ in range(self.num_step):
            p.stepSimulation()

        self.simulation_step = 0

        return ob

    def step(self, action):
        action = action[0]
        self.action_seq.append(action)
        action = self.action_seq[0]

        joint_pendulum = p.getJointState(self.robot_id, 1)[0]
        joint_pendulum_vel = (joint_pendulum - self.prev_joint_pendulum) * self.f
        self.prev_joint_pendulum = joint_pendulum

        ob = np.array([np.cos(joint_pendulum), np.sin(joint_pendulum), joint_pendulum_vel,])
        ob += np.random.normal(0, self.noise_scale, ob.shape)

        rew = 0
        for _ in range(self.num_step):
            p.setJointMotorControl2(self.robot_id, 1, controlMode=p.TORQUE_CONTROL, force=self.max_force*action)
            self.simulation_step += 1
            jp, jp_v = p.getJointState(self.robot_id, 1)[0:2]
            cost = angle_normalize(jp - np.pi)**2
            cost += self.speed_coef * jp_v**2
            rew -= cost # for different Hz
            p.stepSimulation()
        self.action_prev = action

        done = self.simulation_step >= 240 * self.max_seconds

        env_info = {}
        env_info['joint_damping'] = self.joint_damping
        env_info['joint_friction'] = self.joint_friction
        env_info['linear_damping'] = self.linear_damping
        env_info['lateral_friction'] = self.lateral_friction
        env_info['push_force'] = self.push_force
        env_info['mass'] = self.mass
        env_info['f'] = self.f
        env_info['latency'] = self.latency
        env_info['max_force'] = self.max_force

        return ob, rew * self.rew_scale, done, env_info

    def render(self, mode='human'):
        view_mat = p.computeViewMatrix(
            cameraTargetPosition=[0, 0, 0.2],
            cameraEyePosition=[1, 0, 0.2],
            cameraUpVector=[0, 0, 1],
        )
        nearVal = 0.05
        farVal = 100
        proj_mat = p.computeProjectionMatrixFOV(
            fov=45,
            aspect=1,
            nearVal=nearVal,
            farVal=farVal
        )
        (_, _, rgb_px, _, _) = p.getCameraImage(
            width=512, height=512, viewMatrix=view_mat,
            projectionMatrix=proj_mat, renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(rgb_px, 'uint8')
        rgb_array = rgb_array.reshape(512, 512, 4)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array
