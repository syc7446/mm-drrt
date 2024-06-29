import argparse
import random
import numpy as np

from external.pybullet_planning.pybullet_tools.utils import connect, disconnect, set_camera_pose

from examples.envs.pick_place_env import PickPlaceEnvironment, PickPlaceCameraSetup
from examples.envs.object_handover_env import ObjectHandoverEnvironment, ObjectHandoverCameraSetup
from examples.envs.object_cleaning_env import ObjectCleaningEnvironment, ObjectCleaningCameraSetup
from examples.envs.example_single_robot_env import ExampleSingleRobotEnvironment, ExampleSingleRobotCameraSetup
from examples.envs.example_two_robots_env import ExampleTwoRobotsEnvironment, ExampleTwoRobotsCameraSetup
from mm_drrt.planner.task_planner import PlanSkeleton
from experiments.data_saver import data_saver

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_robots', type=int, default=1)
parser.add_argument('--num_objs', type=int, default=1)
parser.add_argument('--num_placement_samples', type=int, default=30)
parser.add_argument('--num_base_samples', type=int, default=50)
parser.add_argument('--num_arm_samples', type=int, default=20)
parser.add_argument('--arm', type=str, default='left')
parser.add_argument('--grasp_type', type=str, default='side')
parser.add_argument('--env_type', type=str, default='exp_single_robot')    # options: pickplace, handover, cleaning, exp_single_robot, exp_two_robots
parser.add_argument('--use_gui', action='store_false')
parser.add_argument('--use_debug', action='store_true')
# dRRT params
parser.add_argument('--drrt_num_iters', type=int, default=10)
parser.add_argument('--drrt_time_limit', type=int, default=2000)

opt = parser.parse_args()
print(opt)

random.seed(opt.seed)
np.random.seed(opt.seed)

sim_id = connect(use_gui=opt.use_gui)
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
elif opt.env_type == 'exp_single_robot':
    set_camera_pose(camera_point=ExampleSingleRobotCameraSetup[0], target_point=ExampleSingleRobotCameraSetup[1])
    env = ExampleSingleRobotEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                        grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)
elif opt.env_type == 'exp_two_robots':
    set_camera_pose(camera_point=ExampleTwoRobotsCameraSetup[0], target_point=ExampleTwoRobotsCameraSetup[1])
    env = ExampleTwoRobotsEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                      grasp_type=opt.grasp_type, sim_id=sim_id, seed=opt.seed)

# input plan skeleton and ordering constraints
plan, action_orders, obj_orders, init_order_constraints = env.create_plan_order_constraints()

assert opt.num_robots == len(action_orders), "Error: num_robots is not properly set"
ps = PlanSkeleton(env, plan, obj_orders, init_order_constraints, opt.num_placement_samples, opt.use_debug)
composite_path = ps.plan_refinement(opt.num_base_samples, opt.num_arm_samples, opt.drrt_num_iters, opt.drrt_time_limit)
# data_saver(composite_path, opt)

disconnect()