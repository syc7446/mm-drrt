import sys
import os
import argparse
import random
import numpy as np

from external.pybullet_planning.pybullet_tools.utils import connect, disconnect, disable_real_time, set_camera_pose

from examples.pick_place_env import PickPlaceEnvironment

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_objs', type=int, default=2)
parser.add_argument('--arm', type=str, default='left')
parser.add_argument('--grasp_type', type=str, default='side')
parser.add_argument('--use_gui', action='store_true')

opt = parser.parse_args()
print(opt)

random.seed(opt.seed)
np.random.seed(opt.seed)

sim_id = connect(use_gui=opt.use_gui)
env = PickPlaceEnvironment(num_objs=opt.num_objs, arm=opt.arm, grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)
env.pick_action(env._objs[0])

disconnect()