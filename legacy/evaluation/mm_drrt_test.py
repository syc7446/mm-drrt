#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import time
import pickle

from external.pybullet_planning.pybullet_tools.utils import connect, disconnect, wait_for_user, create_box, dump_body, \
    get_link_pose, euler_from_quat, GREY, WHITE, set_camera_pose, create_flying_body, create_shape, get_cylinder_geometry, \
    get_movable_joints, get_links, T2, set_joint_positions, set_numpy_seed, \
    load_pybullet, set_quat, set_pose, quat_from_euler, Euler, PI, Pose, Point, \
    add_line, GREEN, intrinsic_euler_from_quat
from mm_drrt.utils.motion_planner_utils import get_max_length_list, set_random_seed
from legacy.evaluation.config import NUM_ROBOTS, ROBOT_COLORS, NUM_SAMPLES_PRM, NUM_DRRT_ITERS, TIME_LIMIT, USE_GUI, \
    USE_DEBUG_PLOT, USE_DEBUG_VERBAL, DEBUG_ROBOT_ID, SIZE_X, SIZE_Y, SLEEP, SEED, CUSTOM_LIMITS, USE_DRRT_STAR, MODE
from legacy.evaluation.drrt_star import dRRTStar
from legacy.evaluation.motion_planner_utils import sub_plan_joint_motion


def main(group=T2):
    set_numpy_seed(SEED)
    set_random_seed(SEED)
    connect(use_gui=USE_GUI)
    set_camera_pose(camera_point=np.array([1., -1., 1.]))
    # TODO: can also create all links and fix some joints
    # TODO: T2(2) motion planner

    wall_right = load_pybullet("examples/models/long_wall.urdf", fixed_base=True)
    wall_left = load_pybullet("examples/models/long_wall.urdf", fixed_base=True)
    wall_top = load_pybullet("examples/models/short_wall.urdf", fixed_base=True)
    wall_bottom = load_pybullet("examples/models/short_wall.urdf", fixed_base=True)
    set_pose(wall_right, Pose(Point(x=SIZE_X)))
    set_pose(wall_left, Pose(Point(x=-SIZE_X)))
    set_pose(wall_top, ((Point(y=SIZE_Y)), quat_from_euler(Euler(yaw=PI / 2))))
    set_pose(wall_bottom, ((Point(y=-SIZE_Y)), quat_from_euler(Euler(yaw=PI / 2))))
    obstacles = [wall_right, wall_left, wall_top, wall_bottom]

    # Initialization
    start_points, goal_points = [], []
    for r in range(NUM_ROBOTS):
        each_start_points, each_goal_points = [], []
        if r == 0:
            # subproblem 0
            each_start_points.append(np.array([0, -.9]))
            each_goal_points.append(np.array([0, .8]))
            # subproblem 1
            each_start_points.append(np.array([0, .8]))
            each_goal_points.append(np.array([0, -.7]))
        elif r == 1:
            # subproblem 0
            each_start_points.append(np.array([0, -.7]))
            each_goal_points.append(np.array([0, .8]))
            # subproblem 1
            each_start_points.append(np.array([0, .8]))
            each_goal_points.append(np.array([0, -.9]))
        assert not np.array_equal(each_start_points[0], each_goal_points[1]), "Currently the same start and goal pair is not allowed."
        start_points.append(each_start_points)
        goal_points.append(each_goal_points)

    # Solve individual problems
    robots, joints, body_links = [], [], []
    roadmaps, heuristic_vals = [], []
    for r in range(NUM_ROBOTS):
        collision_id, visual_id = create_shape(get_cylinder_geometry(radius=0.05, height=0.1), color=ROBOT_COLORS[r])
        robots.append(create_flying_body(group, collision_id, visual_id))

        body_links.append(get_links(robots[r])[-1])
        joints.append(get_movable_joints(robots[r]))
        joint_from_group = dict(zip(group, joints[r]))
        print(joint_from_group)
        # print(get_aabb(robot, body_link))
        dump_body(robots[r], fixed=False)

    if MODE == 'solving':
        for r in range(NUM_ROBOTS):
            each_roadmaps, each_heuristic_vals = [], []
            for subprob in range(len(start_points)):
                custom_limits = {joint_from_group[j]: l for j, l in CUSTOM_LIMITS.items()}

                start_conf = np.concatenate([start_points[r][subprob]])
                goal_conf = np.concatenate([goal_points[r][subprob]])

                set_joint_positions(robots[r], joints[r], start_conf)
                #print(initial_point, get_link_pose(robot, body_link))
                #set_pose(robot, Pose(point=-1.*np.ones(3)))

                # TODO: sample orientation uniformly at random
                # http://planning.cs.uiuc.edu/node198.html
                #from pybullet_tools.transformations import random_quaternion
                roadmap, heuristic_val = sub_plan_joint_motion(robots[r], joints[r], goal_conf,
                                                               obstacles=obstacles, self_collisions=False,
                                                               custom_limits=custom_limits, num_samples=NUM_SAMPLES_PRM,
                                                               use_drrt_star=USE_DRRT_STAR, use_debug_plot=USE_DEBUG_PLOT)
                if USE_DRRT_STAR: each_heuristic_vals.append(heuristic_val)
                each_roadmaps.append(roadmap)
                print('Robot %d''s roadmap for subproblem %d is obtained.' % (r, subprob))
            roadmaps.append(each_roadmaps)
            heuristic_vals.append(each_heuristic_vals)

        # Solve dRRT
        tensor_roadmap = dRRTStar(robots, joints, roadmaps, num_robots=NUM_ROBOTS, heuristic_vals=heuristic_vals)
        start_time = time.time()
        path = tensor_roadmap.grow(num_iters=NUM_DRRT_ITERS, start_time=start_time, time_limit=TIME_LIMIT,
                                   use_debug_plot=USE_DEBUG_PLOT, use_debug_verbal=USE_DEBUG_VERBAL,
                                   debug_robot_id=DEBUG_ROBOT_ID)
        # Save result
        db = {}
        db['robots'] = robots
        db['path'] = path
        db['joints'] = joints
        db['body_links'] = body_links
        file = open('test_multirobot_t2', 'wb')
        pickle.dump(db, file)
    elif MODE == 'visualization':
        file = open('test_multirobot_t2', 'rb')
        db = pickle.load(file)

        for i in range(len(db['path'])):
            for j in range(get_max_length_list(db['path'][i].sub_local_paths)):
                for r in range(NUM_ROBOTS):
                    if len(db['path'][i].sub_local_paths[r]) <= j:
                        set_joint_positions(db['robots'][r], db['joints'][r], db['path'][i].sub_local_paths[r][len(db['path'][i].sub_local_paths[r])-1])
                    else:
                        set_joint_positions(db['robots'][r], db['joints'][r], db['path'][i].sub_local_paths[r][j])
                    point, _ = get_link_pose(db['robots'][r], db['body_links'][r])
                time.sleep(SLEEP)
    disconnect()

if __name__ == '__main__':
    main()
