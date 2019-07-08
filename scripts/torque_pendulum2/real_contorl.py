import argparse

from dynamixel_workbench_msgs.msg import MX2Ext
from dynamixel_workbench_msgs.srv import DynamixelCommand, DynamixelCommandRequest
import numpy as np
import redis
import rospy


class RealControl(object):
    def __init__(self, init_joint_pendulum=0, max_current=1940, baud_rate=57600):
        self.init_joint_pendulum = init_joint_pendulum
        self.max_current = max_current
        self.baud_rate = baud_rate

        self.add_term = np.deg2rad(10)
        self.prev_joint_pendulum_raw = None

        self.r = redis.StrictRedis()

        rospy.init_node('controller_node')
        rospy.wait_for_service('/dynamixel/command')
        self.service = rospy.ServiceProxy('/dynamixel/command', DynamixelCommand)
        self.request = DynamixelCommandRequest()

        # Disable torque (if necessary) for changing the operating mode
        self.send('Torque_Enable', 0)
        rospy.sleep(0.1)

        # Set Baudrate to user-defined value
        #self.send('Baud_Rate', self.baud_rate)
        #rospy.sleep(0.1)

        # Set operating mode to 3
        self.send('Operating_Mode', 3)
        rospy.sleep(0.1)

        # Enable torque for motor to move
        self.send('Torque_Enable', 1)
        rospy.sleep(0.1)

        #Set initial position
        self.send('Profile_Velocity', 30)
        self.init_joint_pendulum = -np.deg2rad(10)%(2*np.pi)
        self.send('Goal_Position', int(self.init_joint_pendulum / np.pi / 2. * 4095))
        rospy.sleep(10)

        # Disable torque for changing the operating mode
        self.send('Torque_Enable', 0)
        rospy.sleep(0.1)

        # Set operating mode to 0
        self.send('Operating_Mode', 0)
        rospy.sleep(0.1)

        # Enable torque for motor to move
        self.send('Torque_Enable', 1)

        self.subscriber = rospy.Subscriber('/dynamixel/MX', MX2Ext, self.callback, queue_size=1)

    def __del__(self):
        self.terminate()

    def terminate(self):
        rospy.loginfo('---- Shutting down the node ----')

        rospy.loginfo('Stop current')
        self.send('Goal_Current', 0)

        #rospy.loginfo('Reset Baud rate to 57600')
        #self.send('Baud_Rate', 57600)

        rospy.signal_shutdown("shutdown")

    def send(self, addr_name, value):
        self.request.command = 'addr'
        self.request.id = 1
        self.request.addr_name = addr_name
        self.request.value = value

        ret = self.service(self.request)
        if not ret.comm_result:
            rospy.loginfo("Sending addr_name: {}, value: {} is failed".format(addr_name, value))
        return ret

    def callback(self, msg):
        self.r.set('start', 'true')
        flag = self.r.get('flag')
        if flag == 'finish':
            self.terminate()
            return

        joint_pendulum_raw = (msg.Present_Position*2*np.pi) / 4095.0
        if self.prev_joint_pendulum_raw is None:
            self.prev_joint_pendulum_raw = joint_pendulum_raw
        if self.prev_joint_pendulum_raw - joint_pendulum_raw > 3.5:
            self.add_term += np.pi*2
        elif self.prev_joint_pendulum_raw - joint_pendulum_raw < -3.5:
            self.add_term -= np.pi*2

        self.prev_joint_pendulum_raw = joint_pendulum_raw
        joint_pendulum = joint_pendulum_raw + self.add_term

        self.r.set('joint_motor', str(joint_pendulum))

        torque = self.r.get('torque')
        if torque is None:
            torque = 0
        torque = int(float(torque) * self.max_current)

        self.send('Goal_Current', torque)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_joint_pendulum', type=float, default=0)
    parser.add_argument('--baud_rate', type=int, default=57600)
    parser.add_argument('--max_current', type=int, default=1940)
    args = parser.parse_args()

    controller = RealControl(args.init_joint_pendulum, args.max_current, args.baud_rate)

    try:
        rospy.spin()
    except:
        del controller

