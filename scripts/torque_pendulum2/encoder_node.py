import argparse

import numpy as np
import redis
import rospy
from std_msgs.msg import Float32

Hz = 25
JOINT_ENCODER_OFFSET = 2.8125 - 180

r = redis.StrictRedis()

prev_raw_data = None
add_term = 0
def callback(data):
    global prev_raw_data
    global add_term
    if prev_raw_data is None:
        prev_raw_data = data.data
    raw_data = data.data
    if prev_raw_data - raw_data > 200:
        add_term += 360
    elif prev_raw_data - raw_data < -200:
        add_term -= 360
    prev_raw_data = raw_data

    #NOTE: negative for sim
    joint_encoder = -np.deg2rad(raw_data + add_term + JOINT_ENCODER_OFFSET)

    r.set('joint_encoder', str(joint_encoder))

rospy.Subscriber('/encoder/state', Float32, callback)
rospy.init_node('encoder_node')
rospy.spin()

