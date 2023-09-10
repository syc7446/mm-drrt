import pickle
import os
import argparse

from external.pybullet_planning.pybullet_tools.utils import join_paths, get_parent_dir, connect, disconnect, \
    disable_real_time, set_joint_positions, joint_from_name, wait_for_duration, set_camera_pose, VideoSaver
from external.pybullet_planning.pybullet_tools.pr2_utils import PR2_GROUPS

from examples.envs.pick_place_env import PickPlaceEnvironment, PickPlaceCameraSetup
from examples.envs.object_handover_env import ObjectHandoverEnvironment, ObjectHandoverCameraSetup
from examples.envs.object_cleaning_env import ObjectCleaningEnvironment, ObjectCleaningCameraSetup

from mm_drrt.utils.motion_planner_utils import get_max_length_list

'''
Make sure to comment out "get_gripper" in pick_place_env.py; otherwise, the initial gripper will remain to the end
'''

parser = argparse.ArgumentParser()
parser.add_argument('--load_file', type=str, required=True, help="file path and name")
parser.add_argument('--env_type', type=str, default='cleaning')    # options: pickplace, handover, cleaning

opt = parser.parse_args()
print(opt)

path = join_paths(get_parent_dir(__file__), os.pardir, '.')
dbfile = open(path+'/experiments/'+opt.load_file, 'rb')
db = pickle.load(dbfile)

composite_path = db['composite_path']
opt = db['opt']

sim_id = connect(use_gui=True)
disable_real_time()
if opt.env_type == 'pickplace':
    set_camera_pose(camera_point=PickPlaceCameraSetup[0], target_point=PickPlaceCameraSetup[1])
    env = PickPlaceEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                               grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)
elif opt.env_type == 'handover':
    set_camera_pose(camera_point=ObjectHandoverCameraSetup[0], target_point=ObjectHandoverCameraSetup[1])
    env = ObjectHandoverEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                    grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)
elif opt.env_type == 'cleaning':
    set_camera_pose(camera_point=ObjectCleaningCameraSetup[0], target_point=ObjectCleaningCameraSetup[1])
    env = ObjectCleaningEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                    grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)

# video_saver = VideoSaver('video.mp4')

for i in range(len(composite_path)):
    for j in range(get_max_length_list(composite_path[i].sub_local_paths)):
        for r in range(opt.num_robots):
            joints = [joint_from_name(env.robots[r], name) for name in PR2_GROUPS['base']] + \
                     [joint_from_name(env.robots[r], name) for name in PR2_GROUPS['left_arm']]
            if len(composite_path[i].sub_local_paths[r]) <= j:
                set_joint_positions(env.robots[r], joints, composite_path[i].sub_local_paths[r][len(composite_path[i].sub_local_paths[r])-1])
            else:
                set_joint_positions(env.robots[r], joints, composite_path[i].sub_local_paths[r][j])
            if composite_path[i].attachments[r]:
                composite_path[i].attachments[r][0].assign()
        wait_for_duration(0.01)

# video_saver.restore()
dbfile.close()