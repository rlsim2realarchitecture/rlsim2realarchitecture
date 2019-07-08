import hashlib
import os
from pathlib import Path

pkg_dir = Path(__file__).parent.absolute()


template_pendulum = """<?xml version="1.0" ?>
<robot name="pendulum" xmlns:xacro="http://wiki.ros.org/xacro">
  <!-- colours based on RAL values given in "FAQ - Colours of robot and robot
       controller", version "KUKA.Tifs | 2010-01-21 |YM| DefaultColorsRobotAndController.doc",
       downloaded 2015-07-18 from
       http://www.kuka.be/main/cservice/faqs/hardware/DefaultColorsRobotAndController.pdf

       all RAL colours converted using http://www.visual-graphics.de/en/customer-care/ral-colours
  -->
  <material name="aluminum">
    <color rgba="0.5 0.5 0.5 1"/>
  </material>
  <link name="base_link">
    <inertial>
      <mass value="9.85208649605"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
      <inertia ixx="1" ixy="0.0" ixz="0.0" iyy="1" iyz="0.0" izz="1"/>
    </inertial>
  </link>
  <joint name="joint_attachment" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0.2"/>
    <parent link="base_link"/>
    <child link="attachment"/>
  </joint>
  <link name="attachment">
    <inertial>
      <mass value="9.85208649605"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0."/>
      <inertia ixx="1" ixy="0.0" ixz="0.0" iyy="1" iyz="0.0" izz="1"/>
    </inertial>
  </link>
  <joint name="joint_pendulum" type="continuous">
    <origin rpy="0 -0 0" xyz="0 0 0"/>
    <parent link="attachment"/>
    <child link="pendulum"/>
    <axis xyz="1 0 0"/>
    <!--limit effort="0" velocity="900"/-->
    <dynamics damping="{joint_damping}" friction="{joint_friction}"/>
  </joint>
  <link name="pendulum">
    <collision>
      <origin rpy="0 0 0" xyz="0 0 -0.1"/>
      <geometry>
        <!--cylinder length="0.2" radius="0.0075"/-->
        <box size="0.015 0.015 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
      <inertia ixx="1" ixy="0.0" ixz="0.0" iyy="1" iyz="0.0" izz="1"/>
    </inertial>
  </link>
  <link name="weight">
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.007 0.007 0.007"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="100"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0"/>
      <inertia ixx="1" ixy="0.0" ixz="0.0" iyy="1" iyz="0.0" izz="1"/>
    </inertial>
  </link>
  <joint name="joint_weight" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 -0.2"/>
    <parent link="pendulum"/>
    <child link="weight"/>
  </joint>

</robot>
"""

def pendulum_from_template(joint_damping, joint_friction):
    (pkg_dir / 'models' / 'pendulum' / 'garbage').mkdir(exist_ok=True, parents=True)
    urdf = template_pendulum.format(joint_damping=joint_damping, joint_friction=joint_friction)
    file_path = pkg_dir / 'models' / 'pendulum' / 'garbage' / (hashlib.sha256(urdf.encode()).hexdigest() + str(os.getpid()) + '.urdf')
    with open(str(file_path), 'w') as f:
        f.write(urdf)
        f.flush()
    return Path(file_path)


template_double_pendulum = """
<robot name="inverted_pendulum" xmlns:xacro="http://wiki.ros.org/xacro">
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>

  <link name="base_link">
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.12 0.07 0.07"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.0384836248495"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
      <inertia ixx="1" ixy="0.0" ixz="0.0"
               iyy="1" iyz="0.0"
               izz="1"/>
    </inertial>
  </link>

  <joint name="joint_encoder" type="continuous">
    <origin rpy="0 0 0" xyz="0.07 0 0.0"/>
    <parent link="base_link"/>
    <child link="pendulum"/>
    <axis xyz="1 0 0"/>
    <limit effort="0" velocity="900"/>
    <dynamics damping="{joint_encoder_damping}"
              friction="{joint_encoder_friction}"/>
  </joint>

  <link name="pendulum">
    <collision>
      <origin rpy="0 0 0" xyz="0 0 -0.1165"/>
      <geometry>
        <cylinder length="0.233" radius="0.005"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 -0.05825"/>
      <inertia ixx="0.0022651667" ixy="0.0" ixz="0.0"
               iyy="0.0022651667" iyz="0.0"
               izz="6.25e-06"/>
    </inertial>
  </link>

  <joint name="joint_fixed_motor" type="fixed">
    <origin rpy="0 0 0" xyz="0 0.025 -0.233"/>
    <parent link="pendulum"/>
    <child link="motor"/>
    <axis xyz="1 0 0"/>
    <limit effort="0" velocity="0"/>
    <dynamics damping="0" friction="0"/>
  </joint>

  <link name="motor">
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.041 0.041 0.061"/>
      </geometry>
      <material name="">
        <color rgba="0 0 0 1"/>
      </material>
    </collision>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.041 0.041 0.061"/>
      </geometry>
      <material name="black" />
    </visual>
    <inertial>
      <mass value="0.5"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
      <inertia ixx="1" ixy="0.0" ixz="0.0" iyy="1" iyz="0.0" izz="1"/>
    </inertial>
  </link>

  <joint name="joint_motor" type="continuous">
    <origin rpy="0 0 0" xyz="0.03 0 0.02"/>
    <parent link="motor"/>
    <child link="pendulum2"/>
    <axis xyz="1 0 0"/>
    <dynamics damping="{joint_motor_damping}"
              friction="{joint_motor_friction}"/>
  </joint>

  <link name="pendulum2">
    <collision>
      <origin rpy="0 0 0" xyz="0 0 -0.125"/>
      <geometry>
        <cylinder length="0.25" radius="0.005"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 -0.0625"/>
      <inertia ixx="0.0002764478849712759" ixy="0.0" ixz="0.0"
               iyy="0.0002764478849712759" iyz="0.0"
               izz="6.626797244280169e-07"/>
    </inertial>
  </link>

  <joint name="joint_weight" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="pendulum2"/>
    <child link="weight"/>
  </joint>

  <link name="weight">
    <collision>
      <origin rpy="0 0 0" xyz="0 0 -0.25"/>
      <geometry>
        <box size="0.01 0.01 0.01"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 -0.125"/>
      <inertia ixx="1" ixy="0.0" ixz="0.0"
               iyy="1" iyz="0.0"
               izz="1"/>
    </inertial>
  </link>

</robot>
"""


def double_pendulum_from_template(
        joint_encoder_damping, joint_encoder_friction,
        joint_motor_damping, joint_motor_friction,
        mass):
    (pkg_dir / 'models' / 'pendulum' / 'garbage').mkdir(
        exist_ok=True, parents=True)
    urdf = template_double_pendulum.format(
        joint_encoder_damping=joint_encoder_damping,
        joint_encoder_friction=joint_encoder_friction,
        joint_motor_damping=joint_motor_damping,
        joint_motor_friction=joint_motor_friction,
        mass=mass)
    file_path = pkg_dir / 'models' / 'pendulum' / 'garbage' / \
        (hashlib.sha256(urdf.encode()).hexdigest() +
         str(os.getpid()) + '.urdf')
    with open(str(file_path), 'w') as f:
        f.write(urdf)
        f.flush()
    return Path(file_path)
