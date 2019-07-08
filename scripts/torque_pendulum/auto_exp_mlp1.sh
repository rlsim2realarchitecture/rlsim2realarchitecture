#!/bin/bash

echo mlp1
source ~/catkin_ws/torque_pendulum/devel/setup.bash
python ./real_contorl.py --max_current $2 &
pid=$!
source ~/.virtuals/py3/bin/activate
python ./inference_node_mlp.py --log $3 --pol $1 --timestep 1
kill -9 $pid


