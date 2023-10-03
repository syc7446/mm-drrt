import pickle
import os
import argparse
import random
import numpy as np
from datetime import datetime

from external.pybullet_planning.pybullet_tools.utils import join_paths, get_parent_dir, connect, disconnect, \
    disable_real_time, set_joint_positions, joint_from_name, wait_for_duration, set_camera_pose, VideoSaver, get_default_resolution
from external.pybullet_planning.pybullet_tools.pr2_utils import PR2_GROUPS, get_arm_joints, get_disabled_collisions

from examples.envs.pick_place_env import PickPlaceEnvironment, PickPlaceCameraSetup
from examples.envs.object_handover_env import ObjectHandoverEnvironment, ObjectHandoverCameraSetup
from examples.envs.object_cleaning_env import ObjectCleaningEnvironment, ObjectCleaningCameraSetup

from mm_drrt.utils.motion_planner_utils import get_max_length_list, get_smoothing_fn, smooth_path

'''
Make sure to comment out "get_gripper" in pick_place_env.py; otherwise, the initial gripper will remain to the end
'''

parser = argparse.ArgumentParser()
parser.add_argument('--load_file', type=str, required=True, help="file path and name")
parser.add_argument('--env_type', type=str, default='cleaning')    # options: pickplace, handover, cleaning
parser.add_argument('--smoothing_max_iterations', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)

fopt = parser.parse_args()
print(fopt)

random.seed(fopt.seed)
np.random.seed(fopt.seed)

path = join_paths(get_parent_dir(__file__), os.pardir, '.')
dbfile = open(path+'/experiments/'+fopt.load_file, 'rb')
db = pickle.load(dbfile)

composite_path = db['composite_path']
opt = db['opt']

sim_id = connect(use_gui=False)
disable_real_time()
if fopt.env_type == 'pickplace':
    set_camera_pose(camera_point=PickPlaceCameraSetup[0], target_point=PickPlaceCameraSetup[1])
    env = PickPlaceEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                               grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)
elif fopt.env_type == 'handover':
    set_camera_pose(camera_point=ObjectHandoverCameraSetup[0], target_point=ObjectHandoverCameraSetup[1])
    env = ObjectHandoverEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                    grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)
elif fopt.env_type == 'cleaning':
    set_camera_pose(camera_point=ObjectCleaningCameraSetup[0], target_point=ObjectCleaningCameraSetup[1])
    env = ObjectCleaningEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                    grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)


# preprocessing for smoothing
paths, attachments = [], []
cur_subprob_id = [0 for _ in range(opt.num_robots)]
path = []
for i in range(len(composite_path)):
    for j in range(get_max_length_list(composite_path[i].sub_local_paths)):
        q = ()
        for r in range(opt.num_robots):
            if len(composite_path[i].sub_local_paths[r]) <= j:
                q += composite_path[i].sub_local_paths[r][len(composite_path[i].sub_local_paths[r])-1]
            else:
                q +=  composite_path[i].sub_local_paths[r][j]
        path.append(q)
    if i == len(composite_path) - 1:
        paths.append(path)
        attachments.append(composite_path[i].attachments)
    if cur_subprob_id != composite_path[i].subprob_id:
        cur_subprob_id = composite_path[i].subprob_id
        paths.append(path)
        attachments.append(composite_path[i].attachments)
        path = []


# generate sub_smoothing functions
sub_sample_fns, sub_distance_fns, sub_extend_fns, sub_collision_fns = [], [], [], []
obstacles = env.fixed_obstacles
for robot in env.robots.values():
    joints = [joint_from_name(robot, name) for name in PR2_GROUPS['base']] + list(get_arm_joints(robot, env._arm))
    disabled_collisions = get_disabled_collisions(robot)
    resolutions = np.concatenate((np.array([2 * get_default_resolution(robot, 2), 2 * get_default_resolution(robot, 2),
                                            get_default_resolution(robot, 2)]),
                                  0.05 ** np.ones(len(get_arm_joints(robot, env._arm)))), axis=0)
    custom_limits = env.custom_limits[robot]
    sub_sample_fn, sub_distance_fn, sub_extend_fn, sub_collision_fn = get_smoothing_fn(robot, joints, obstacles=obstacles,
                                                                                       self_collisions=False, disabled_collisions=disabled_collisions,
                                                                                       resolutions=resolutions, custom_limits=custom_limits)
    sub_sample_fns.append(sub_sample_fn)
    sub_distance_fns.append(sub_distance_fn)
    sub_extend_fns.append(sub_extend_fn)
    sub_collision_fns.append(sub_collision_fn)


# smoothed path
smoothed_paths = []
for i, path in enumerate(paths):
    print('path ', i)
    smoothed_paths.append(smooth_path(path, robots=list(env.robots.values()), max_iterations=fopt.smoothing_max_iterations,
                                      sub_extend_fns=sub_extend_fns, sub_collision_fns=sub_collision_fns, sub_distance_fns=sub_distance_fns))

dbfile = open('./experiments/run_' + str(opt.env_type) + '_smooth_{}'.format(datetime.now()), 'ab')
db = {}
db['smoothed_paths'] = smoothed_paths
db['attachments'] = attachments
db['opt'] = opt
pickle.dump(db, dbfile)
dbfile.close()
