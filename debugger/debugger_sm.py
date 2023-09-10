import argparse
import random
import numpy as np
import pickle
import time

from external.pybullet_planning.pybullet_tools.utils import connect, disconnect, set_joint_positions

from examples.envs.pick_place_env import PickPlaceEnvironment
from debugger.utils import sub_plan_joint_motion
from debugger.drrt_star import dRRTStar

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_robots', type=int, default=2)
parser.add_argument('--num_objs', type=int, default=2)
parser.add_argument('--num_placement_samples', type=int, default=10)
parser.add_argument('--num_base_samples', type=int, default=25)
parser.add_argument('--num_arm_samples', type=int, default=20)
parser.add_argument('--arm', type=str, default='left')
parser.add_argument('--grasp_type', type=str, default='side')
parser.add_argument('--use_gui', action='store_true')
parser.add_argument('--use_debug', action='store_true')
# dRRT params
parser.add_argument('--drrt_num_iters', type=int, default=10)
parser.add_argument('--drrt_time_limit', type=int, default=2000)

opt = parser.parse_args()
print(opt)

random.seed(opt.seed)
np.random.seed(opt.seed)

sim_id = connect(use_gui=opt.use_gui)
env = PickPlaceEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                           grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)

initial_conf = [(-0.3187382163228052, -0.4015353053923896, -0.8963159896307984, 0.39277395, 0.33330058, 0.0, -1.52238431, 2.72170996, -1.21946936, -2.98914779), \
               (-0.5253967642784119, 0.9448966383934021, 0.7533274800902596, 0.39277395, 0.33330058, 0.0, -1.52238431, 2.72170996, -1.21946936, -2.98914779)]
roadmaps, heuristic_vals, final_conf = [], [], []
robots = [9, 10]
joints = [[0, 1, 2, 61, 62, 63, 65, 66, 68, 69], [0, 1, 2, 61, 62, 63, 65, 66, 68, 69]]
obstacles = [8, 5, 2, 3, 4]

# Save result
# db = {}
# file = open('test_multirobot_t2', 'wb')
# pickle.dump(db, file)

file = open('data_sm_2', 'rb')
db = pickle.load(file)
roadmaps.append(db['roadmap0'])
roadmaps.append(db['roadmap1'])
heuristic_vals.append(db['roadmap0:heuristic_vals'])
heuristic_vals.append(db['roadmap1:heuristic_vals'])
final_conf.append(db['roadmap0:final_config'])
final_conf.append(db['roadmap1:final_config'])
file.close()

for r in range(len(robots)):
    roadmaps[r] = sub_plan_joint_motion(roadmaps[r], robots[r], joints[r], initial_conf[r], final_conf[r], obstacles, custom_limits=env.custom_limits)

tensor_roadmap = dRRTStar(robots, joints, roadmaps, num_robots=2, heuristic_vals=heuristic_vals)
tensor_roadmap.debug_neighbors()
path = tensor_roadmap.grow(num_iters=opt.drrt_num_iters, start_time=time.time(), time_limit=opt.drrt_time_limit)

disconnect()

# save roadmap data to pickle
# import pickle
# db = {}
# file = open('test_multirobot_t2', 'wb')
# for r in range(2):
#     self.roadmaps[r][6].sub_sample_fn = None
#     self.roadmaps[r][6].sub_distance_fn = None
#     self.roadmaps[r][6].sub_extend_fn = None
#     self.roadmaps[r][6].sub_collision_fn = None
#     self.roadmaps[r][6].distance_fn = None
#     self.roadmaps[r][6].extend_fn = None
#     self.roadmaps[r][6].collision_fn = None
#     db['roadmap'+str(r)] = self.roadmaps[r][6]
#     db['roadmap'+str(r)+':heuristic_vals'] = self.heuristic_vals[r]
#     db['roadmap' + str(r) + ':final_config'] = self.roadmaps[r][6].final_conf
# pickle.dump(db, file)
