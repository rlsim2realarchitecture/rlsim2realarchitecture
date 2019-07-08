from collections import deque
import time

from gym.core import Env
from gym.spaces import Box
import numpy as np
import pybullet as p

from meta_rl_tools.envs.sim2real_envs.data import double_pendulum_from_template
from meta_rl_tools.envs.sim2real_envs.utils import draw
from meta_rl_tools.envs.sim2real_envs.utils import flush
import pdb


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


def joint_name_to_id(object_id, physics_clientid=0):
    joint_num = p.getNumJoints(
        object_id,
        physicsClientId=physics_clientid)
    joint_name_to_joint_id = {}
    for i in range(joint_num):
        joint_name = p.getJointInfo(
            object_id, i,
            physicsClientId=physics_clientid)[1]
        joint_name_to_joint_id[joint_name.decode('utf-8')] = i
    return joint_name_to_joint_id


class TorquePendulum2(Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 25}

    def __init__(
        self,
        goal_deg=5,
        mass_range=(0.05, 0.5),
        # randomized params for dynamics
        f_candidates=(25,),
        latency_range=(1, 6),
        max_force_range=(2, 10),  # randomized params for control
        max_seconds=20,
        noise_scale=0.01,
        render=False,
        debug=False):

        self.goal_deg = goal_deg

        self.mass_range = mass_range
        self.debug = debug

        self.f_candidates = f_candidates
        self.latency_range = latency_range
        self.max_force_range = max_force_range

        self.max_seconds = max_seconds
        self.noise_scale = noise_scale

        if render:
            self.connect = p.GUI
        else:
            self.connect = p.DIRECT
        p.connect(self.connect)

        high = np.ones(1, dtype=np.float32)
        self.action_space = Box(low=-high, high=high, dtype=np.float32)
        high = np.inf * np.ones(6)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)
        self.urdf_path = None

    def __del__(self):
        if self.urdf_path is not None:
            self.urdf_path.unlink()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.resetDebugVisualizerCamera(0.8, 90, 0, [0, 0, 0.0])
        self.joint_encoder_damping  = 0.005                     
        self.joint_motor_damping  = 0.2                         
        self.joint_encoder_friction = 1                         
        self.joint_motor_friction  = 5                          
        self.linear_damping_pendulum2 = 0.04                    
        self.linear_damping_weight = 0.04                       
        self.mass_pendulum2 = 0.1                               
        self.mass_motor = 0.2                                   
        self.mass_weight = np.random.uniform(self.mass_range[0], self.mass_range[1])

        if self.urdf_path is not None:
            self.urdf_path.unlink()
        self.urdf_path = urdf_path = double_pendulum_from_template(
            self.joint_encoder_damping, self.joint_encoder_friction,
            self.joint_motor_damping, self.joint_motor_friction, mass = 1)

        self.robot_id = p.loadURDF(str(urdf_path), [0, 0, 0],
                                   useFixedBase=True)
        self.joint_name_to_id = joint_name_to_id(self.robot_id)
        self.joint_encoder_id = self.joint_name_to_id['joint_encoder']
        self.joint_motor_id = self.joint_name_to_id['joint_motor']
        self.joint_weight_id = self.joint_name_to_id['joint_weight']
        self.joint_fixed_motor_id = self.joint_name_to_id['joint_fixed_motor']

        # deactivate torque
        p.setJointMotorControl2(
            self.robot_id,
            self.joint_encoder_id,
            controlMode=p.VELOCITY_CONTROL, force=0
        )
        p.changeDynamics(self.robot_id, self.joint_encoder_id,
                         linearDamping=0.1,
                         lateralFriction=0.1)

        p.setJointMotorControl2(
            self.robot_id,
            self.joint_motor_id,
            controlMode=p.VELOCITY_CONTROL, force=0
        )

        init_joint_motor = np.random.uniform(-np.pi / 8, np.pi / 8)
        p.resetJointState(self.robot_id,
                          self.joint_encoder_id,
                          init_joint_motor)
        p.resetJointState(self.robot_id,
                          self.joint_motor_id,
                          init_joint_motor)

        # weight
        # http://emanual.robotis.com/docs/en/dxl/mx/mx-64-2/
        #Change dynamics of motor
        p.changeDynamics(
            self.robot_id,
            self.joint_fixed_motor_id,
            mass = self.mass_motor)
        
        #Change dynamics of pendulum2
        p.changeDynamics(
            self.robot_id,
            self.joint_motor_id,
            #This might be a mistake, this code changes the mass of pendulum2, not the motor
            mass=self.mass_pendulum2,
            linearDamping=self.linear_damping_pendulum2,
            lateralFriction=0.1)
        
        #Change dynamics of terminal weight
        p.changeDynamics(
            self.robot_id,
            self.joint_weight_id,
            mass=self.mass_weight,
            linearDamping=self.linear_damping_weight)

        self.f = np.random.choice(self.f_candidates)
        self.num_step = int(240 / self.f)

        self.prev_joint_motor = joint_motor = p.getJointState(
            self.robot_id,
            self.joint_motor_id)[0]
        joint_motor_vel = 0
        self.prev_joint_encoder = joint_encoder = p.getJointState(
            self.robot_id,
            self.joint_encoder_id)[0]
        joint_encoder_vel = 0

        self.latency = np.random.randint(self.latency_range[0],
                                         self.latency_range[1])
        self.action_seq = deque(maxlen=self.latency + 1)
        for _ in range(self.latency + 1):
            self.action_seq.append(0)

        self.max_force = np.random.uniform(
            self.max_force_range[0], self.max_force_range[1]
        )

        ob = np.array(
            [np.cos(joint_motor),
             np.sin(joint_motor),
             joint_motor_vel,
             np.cos(joint_encoder),
             np.sin(joint_encoder),
             joint_encoder_vel], dtype=np.float32
        )
        self.joint_encoder = joint_encoder
        ob += np.random.normal(0, self.noise_scale, ob.shape)

        for _ in range(self.num_step):
            p.stepSimulation()
            if self.debug:
                time.sleep(1 / 240.0)

        self.simulation_step = 0

        return ob

    def step(self, action):
        """Step simulation

        Parameters
        ----------
        action : numpy.ndarray
            input torque of motor. shape is (1, ).

        Returns
        -------
        ob : numpy.ndarray
            observation. shape is (6, ).
            [cos(theta_motor), sin(theta_motor), velocity_motory,
             cos(theta_encoder), sin(theta_encoder), velocity_encoder]
        rew : float
            reward
        done : bool
            if this env's episode is terminated, return True.
        """
        action = action[0]
        self.action_seq.append(action)
        action = self.action_seq[0]

        joint_motor, _, _, _ = p.getJointState(
            self.robot_id, self.joint_motor_id)
        joint_motor_vel = (joint_motor - self.prev_joint_motor) \
            * self.f
        self.prev_joint_motor = joint_motor

        joint_encoder, _ , _, _ = p.getJointState(
            self.robot_id, self.joint_encoder_id)
        self.joint_encoder = joint_encoder
        joint_encoder_vel = (joint_encoder - self.prev_joint_encoder) \
            * self.f
        self.prev_joint_encoder = joint_encoder

        ob = np.array(
            [np.cos(joint_motor),
             np.sin(joint_motor),
             joint_motor_vel,
             np.cos(joint_encoder),
             np.sin(joint_encoder),
             joint_encoder_vel], dtype=np.float32
        )
        ob += np.random.normal(0, self.noise_scale, ob.shape)

        done = False
        for _ in range(self.num_step):
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_motor_id,
                controlMode=p.TORQUE_CONTROL,
                force=self.max_force * action,
            )
            self.simulation_step += 1
            if np.abs(angle_normalize(p.getJointState(self.robot_id, self.joint_encoder_id)[0]) - np.pi) < np.abs(np.deg2rad(self.goal_deg)):
                done = True
            p.stepSimulation()
            if self.debug:
                time.sleep(1 / 240.0)

        if done:
            rew = 1
        else:
            rew = -1

        done |= self.simulation_step >= 240 * self.max_seconds

        env_info = {}
        env_info["joint_encoder_damping"] = self.joint_encoder_damping
        env_info["joint_encoder_friction"] = self.joint_encoder_friction
        env_info["joint_motor_damping"] = self.joint_motor_damping
        env_info["joint_motor_friction"] = self.joint_motor_friction
        env_info["linear_damping_weight"] = self.linear_damping_weight
        env_info["linear_damping_pendulum2"] = self.linear_damping_pendulum2
        env_info["mass_motor"] = self.mass_motor
        env_info["mass_pendulum2"] = self.mass_pendulum2
        env_info["f"] = self.f
        env_info["latency"] = self.latency
        env_info["max_force"] = self.max_force

        return ob, rew, done, env_info

    def render(self, mode="human"):
        view_mat = p.computeViewMatrix(
            cameraTargetPosition=[0, 0, 0.2],
            cameraEyePosition=[2, 0, 0.2],
            cameraUpVector=[0, 0, 1],
        )
        nearVal = 0.05
        farVal = 100
        proj_mat = p.computeProjectionMatrixFOV(
            fov=45, aspect=1, nearVal=nearVal, farVal=farVal
        )
        (_, _, rgb_px, _, _) = p.getCameraImage(
            width=512,
            height=512,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.array(rgb_px, "uint8")
        rgb_array = rgb_array.reshape(512, 512, 4)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array


