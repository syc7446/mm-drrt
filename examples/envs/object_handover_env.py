#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import time
import copy
import random
import operator

from external.pybullet_planning.pybullet_tools.utils import get_pose, get_joint_positions, joints_from_names, is_placement, \
    set_base_values, load_pybullet, create_box, set_point, set_euler, set_quat, set_joint_positions, WorldSaver, \
    set_joint_position, remove_body, BROWN, RED, BLUE, WHITE, GREEN, YELLOW, RGBA
from external.pybullet_planning.pybullet_tools.pr2_utils import get_other_arm, get_carry_conf, set_arm_conf, open_arm, \
    PR2_GROUPS, arm_conf, close_arm, pairwise_collision, get_group_joints, REST_LEFT_ARM
from external.pybullet_planning.pybullet_tools.pr2_primitives import Pose, Conf
from external.pybullet_planning.pybullet_tools.pr2_problems import create_floor, TABLE_MAX_Z

from examples.utils import SINGLE_ROOM, NARROW_TABLE, TABLE, create_shelf, create_shelf_placement, create_pr2, \
    TABLE_POSE_X, TABLE_POSE_Y, SHELF_LENGTH, SHELF_WIDTH, SHELF_HEIGHT, SHELF_REACHABLE_MARGIN, Environment, \
    SINGLE_BIG_ROOM, TWO_ROOMS
from mm_drrt.utils.motion_planner_utils import get_ik_ir_gen, get_ik_fn, get_stable_gen, get_placement_gen, base_motion, \
    get_grasp_gen, Problem, get_gripper, get_configuration, get_arm_positions, arm_retrieval_motion, get_arm_motion_fn


BOX = (.07, .05, .15)
ObjectHandoverCameraSetup = [(-2.5, -2.3, 4.2), (-2.5, -1.3, 0)]

class ObjectHandoverEnvironment(Environment):
    def __init__(self, num_robots, num_objs, arm, grasp_type, sim_id, seed):
        super().__init__(num_objs, seed)
        self._num_robots = num_robots
        self._num_m_objs = num_objs
        self._sim_id = sim_id
        self._objs_to_obj_ids = {}
        self._grippers = {}
        self._grasp_type = grasp_type
        self._initialize(arm)

    def _initialize(self, arm):
        self._arm = arm
        self._create_problem()
        self.grasp_gen_fn = get_grasp_gen(self._grasp_type, collisions=True)
        self.placement_gen_fn = get_placement_gen()
        # for i, obj in enumerate(self._objs):
        #     self._objs_to_obj_ids[obj] = self._problem.movable[i]

    def placement_sample(self, m_objs, f_obj, num_samples):
        placement_gen = self.placement_gen_fn(m_objs, f_obj)
        placements = []
        for _ in range(num_samples):
            (p,) = next(placement_gen)
            placements.append(p)
        return placements

    def is_placement_collision(self, objs_poses, stationary_m_objs,
                               remove_m_objs, add_m_objs, remove_then_add_m_objs, add_then_remove_m_objs):
        for p in objs_poses:
            p.assign()

        # collisions among movable objs that will be added but not removed in the end
        for i in range(len(remove_then_add_m_objs + add_m_objs) - 1):
            for j in range(i + 1, len(remove_then_add_m_objs + add_m_objs)):
                if pairwise_collision((remove_then_add_m_objs + add_m_objs)[i],
                                      (remove_then_add_m_objs + add_m_objs)[j]):
                    return False, []  # is_feasible, collisions

        # collisions with stationary objs
        for add_m_obj in remove_then_add_m_objs + add_then_remove_m_objs + add_m_objs:
            if any(pairwise_collision(add_m_obj, obst) for obst in stationary_m_objs):
                return False, []

        # collisions with objs that will be removed
        collisions = []
        for add_m_obj in remove_then_add_m_objs + add_m_objs:
            for remove_m_obj in remove_m_objs + add_then_remove_m_objs:
                if pairwise_collision(add_m_obj, remove_m_obj):
                    collisions.append((add_m_obj, remove_m_obj))    # (next, pre)
        return True, collisions

    def placement_collision_with_remove_then_add_m_objs(self, init_obj_pose, cur_obj_pose,
                                                        remove_then_add_m_obj_id, remove_then_add_m_obj,
                                                        init_collisions, add_m_objs, add_then_remove_m_objs,
                                                        remove_then_add_m_objs):
        init_obj_pose.assign()
        for id, add_m_obj in enumerate(remove_then_add_m_objs + add_then_remove_m_objs + add_m_objs):
            if id == remove_then_add_m_obj_id:
                continue
            if pairwise_collision(add_m_obj, remove_then_add_m_obj):
                init_collisions.append((add_m_obj, remove_then_add_m_obj)) # (next, pre)
        cur_obj_pose.assign()
        return init_collisions

    def subgoal_sampling(self, robot, obj_orders, actions, action, m_obj, start_obstacles, goal_obstacles, custom_limits={}, use_debug=False):
        # only used for subgoal checking by setting teleport to true
        start_ik_ir_fn = get_ik_ir_gen(robot, self._grippers[robot], max_attempts=25,
                                       collision_objs=start_obstacles['objs'] + self.fixed_obstacles,
                                       custom_limits=custom_limits[robot], use_debug=use_debug)
        goal_ik_ir_fn = get_ik_ir_gen(robot, self._grippers[robot], max_attempts=25,
                                      collision_objs=goal_obstacles['objs'] + self.fixed_obstacles,
                                      custom_limits=custom_limits[robot], use_debug=use_debug)
        self._assign_target_obj_pose(actions, obj_orders, m_obj, action)
        start_obj_pose = Pose(m_obj)
        grasps = list(self.grasp_gen_fn(robot, m_obj))

        for grasp in grasps:
            (g,) = grasp
            for obst_pose in start_obstacles['poses']:
                obst_pose.assign()
            pick_output = next(start_ik_ir_fn(self._arm, m_obj, start_obj_pose, g), None)  # [0]: base pose, [1]: command, [2]: attachment, [3]: gripper_pose
            if pick_output:
                if pick_output[-1]:
                    for obst_pose in goal_obstacles['poses']:
                        obst_pose.assign()
                    place_output = next(goal_ik_ir_fn(self._arm, m_obj, action.goal['place'], g), None)
                    if place_output:
                        if place_output[-1]:
                            action.start['grasp'] = g
                            action.start['place'] = start_obj_pose
                            action.start['base'] = pick_output[0]
                            action.start['approach_conf'] = pick_output[1]
                            action.start['grasp_conf'] = pick_output[2]
                            action.goal['grasp'] = g
                            action.goal['base'] = place_output[0]
                            action.goal['approach_conf'] = place_output[1]
                            action.goal['grasp_conf'] = place_output[2]
                            return True
        return False

    def compute_path(self, robot, action, m_obj, num_base_samples, num_arm_samples, type=None, use_debug=False):
        if type != 'return':
            start_obstacles = action.obstacles['start']
            goal_obstacles = action.obstacles['goal']
            obstacles = set(start_obstacles['objs']) | set(goal_obstacles['objs'])
            for obst_pose in start_obstacles['poses']:
                obst_pose.assign()
            if type == 'transfer':
                copied_objs, copied_objs_poses = [], []
                for id in range(len(goal_obstacles['objs'])):
                    if goal_obstacles['objs'][id] in set(start_obstacles['objs']) & set(goal_obstacles['objs']):
                        copied_objs.append(create_box(BOX[0], BOX[1], BOX[2]))
                        copied_objs_poses.append(goal_obstacles['poses'][id].value)
                        continue
                for id in range(len(copied_objs)):
                    Pose(copied_objs[id], copied_objs_poses[id], action.to_f_obj).assign()
                obstacles = obstacles | set(copied_objs)
            obstacles = list(obstacles) + self.fixed_obstacles
            # obstacles = list(obstacles)
            obj_pose = Pose(m_obj)

            r_expand_configs = get_arm_positions(robot=robot, arm=self._arm)
            if type == 'transit':  # pick
                action.start['place'].assign()  # assign target object's start pose
                base_roadmap, base_heuristic_val = base_motion(robot, action.start['base'].values, action.goal['base'].values,
                                                               obstacles=obstacles, custom_limits=self.custom_limits[robot],
                                                               num_samples=num_base_samples, expand_type='arm', expand_configs=r_expand_configs, use_debug=use_debug)
            elif type == 'transfer':  # place
                attachment = action.start['grasp'].get_attachment(robot, self._arm)
                base_roadmap, base_heuristic_val = base_motion(robot, action.start['base'].values, action.goal['base'].values,
                                                               obstacles=obstacles, custom_limits=self.custom_limits[robot], attachments=[attachment],
                                                               num_samples=num_base_samples, expand_type='arm', expand_configs=r_expand_configs, use_debug=use_debug)

            arm_motion_fn = get_arm_motion_fn(robot, collision_objs=obstacles, num_samples=num_arm_samples,
                                              expand_type='base', expand_configs=action.goal['base'].values, use_debug=use_debug)

            if base_roadmap:
                max_attempts = 5   # TODO: hardcoded max_attempts
                if type == 'transit':   # pick
                    attempts = 0
                    while True:
                        arm_approach_roadmap, arm_approach_heuristic_val = arm_motion_fn(self._arm, m_obj, action.goal['grasp'],
                                                                                         action.goal['approach_conf'], action.goal['grasp_conf'])
                        if arm_approach_roadmap is not None: break
                        else:
                            attempts += 1
                            if attempts >= max_attempts:
                                return False, [], []
                            continue
                    attachment = action.goal['grasp'].get_attachment(robot, self._arm)
                    attempts = 0
                    while True:
                        arm_retrieval_roadmap, arm_retrieval_heuristic_val = arm_retrieval_motion(robot, self._arm, type,
                                                                                                  grasp=action.goal['grasp'],
                                                                                                  start=arm_approach_roadmap.final_conf[len(action.goal['base'].values):],
                                                                                                  goal=tuple(get_carry_conf(self._arm, self._grasp_type)),
                                                                                                  obstacles=obstacles, attachments=[attachment], num_samples=num_arm_samples,
                                                                                                  expand_type='base', expand_configs=action.goal['base'].values,
                                                                                                  use_debug=use_debug)
                        if arm_retrieval_roadmap is not None: break
                        else:
                            attempts += 1
                            if attempts >= max_attempts:
                                return False, [], []
                            continue
                elif type == 'transfer':    # place
                    attachment = action.goal['grasp'].get_attachment(robot, self._arm)
                    attempts = 0
                    while True:
                        arm_approach_roadmap, arm_approach_heuristic_val = arm_motion_fn(self._arm, m_obj, action.goal['grasp'],
                                                                                         action.goal['approach_conf'], action.goal['grasp_conf'],
                                                                                         attachments=[attachment])
                        if arm_approach_roadmap is not None: break
                        else:
                            attempts += 1
                            if attempts >= max_attempts:
                                for copied_obj in copied_objs:
                                    remove_body(copied_obj)
                                return False, [], []
                            continue
                    attempts = 0
                    while True:
                        arm_retrieval_roadmap, arm_retrieval_heuristic_val = arm_retrieval_motion(robot, self._arm, type,
                                                                                                  grasp=action.goal['grasp'],
                                                                                                  start=arm_approach_roadmap.final_conf[len(action.goal['base'].values):],
                                                                                                  goal=tuple(get_carry_conf(self._arm, self._grasp_type)),
                                                                                                  obstacles=obstacles, num_samples=num_arm_samples,
                                                                                                  expand_type='base', expand_configs=action.goal['base'].values,
                                                                                                  use_debug=use_debug)
                        if arm_retrieval_roadmap is not None: break
                        else:
                            attempts += 1
                            if attempts >= max_attempts:
                                for copied_obj in copied_objs:
                                    remove_body(copied_obj)
                                return False, [], []
                            continue
                if type == 'transfer':
                    for copied_obj in copied_objs:
                        remove_body(copied_obj)
                return True, [base_roadmap, arm_approach_roadmap, arm_retrieval_roadmap], \
                       [base_heuristic_val, arm_approach_heuristic_val, arm_retrieval_heuristic_val]
            else:
                if type == 'transfer':
                    for copied_obj in copied_objs:
                        remove_body(copied_obj)
                return False, [], []
        else:   # if type == 'return'
            r_expand_configs = get_arm_positions(robot=robot, arm=self._arm)
            base_roadmap, base_heuristic_val = base_motion(robot, action.start['base'].values,
                                                           action.goal['base'].values,
                                                           custom_limits=self.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=use_debug)
            if base_roadmap:
                return True, [base_roadmap], [base_heuristic_val]
            else:
                return False, [], []


    def get_joints(self, robots):
        joints = []
        for r in robots:
            joints.append(list(get_group_joints(r, 'base')) + list(get_group_joints(r, self._arm + '_arm')))
        return joints

    def get_base_conf(self, robot):
        return Conf(robot, get_group_joints(robot, 'base'), get_configuration(robot, 'base'))

    def get_init_base_conf(self, robot):
        return Conf(robot, get_group_joints(robot, 'base'), self.robots_init_poses[robot])

    def save_world(self):
        return WorldSaver()

    def restore_world(self, saved_world):
        saved_world.restore()

    def _assign_target_obj_pose(self, actions, obj_orders, m_obj, action):
        for a in actions:
            if actions[a] == action:
                    for id in range(len(obj_orders)):
                        if obj_orders[id] == a:
                            if id == 0:
                                self.m_objs_init_placements[m_obj].assign()
                            else:
                                actions[obj_orders[id - 1]].goal['place'].assign()

    def create_plan_order_constraints(self):
        plan = {'a0': ('transit', self.robots[0], self.m_objs[0], None, self.f_objs[0]),
                'a1': ('transfer', self.robots[0], self.m_objs[0], self.f_objs[0], self.f_objs[1]),
                'a2': ('transit', self.robots[0], self.m_objs[1], self.f_objs[1], self.f_objs[0]),
                'a3': ('transfer', self.robots[0], self.m_objs[1], self.f_objs[0], self.f_objs[1]),
                'a4': ('return', self.robots[0], None, None, None),
                'a5': ('transit', self.robots[1], self.m_objs[2], None, self.f_objs[0]),
                'a6': ('transfer', self.robots[1], self.m_objs[2], self.f_objs[0], self.f_objs[1]),
                'a7': ('transit', self.robots[1], self.m_objs[3], self.f_objs[1], self.f_objs[0]),
                'a8': ('transfer', self.robots[1], self.m_objs[3], self.f_objs[0], self.f_objs[1]),
                'a9': ('return', self.robots[1], None, None, None),
                'a10': ('transit', self.robots[2], self.m_objs[0], None, self.f_objs[1]),
                'a11': ('transfer', self.robots[2], self.m_objs[0], self.f_objs[1], self.f_objs[2]),
                'a12': ('transit', self.robots[2], self.m_objs[1], self.f_objs[2], self.f_objs[1]),
                'a13': ('transfer', self.robots[2], self.m_objs[1], self.f_objs[1], self.f_objs[2]),
                'a14': ('return', self.robots[2], None, None, None),
                'a15': ('transit', self.robots[3], self.m_objs[2], None, self.f_objs[1]),
                'a16': ('transfer', self.robots[3], self.m_objs[2], self.f_objs[1], self.f_objs[2]),
                'a17': ('transit', self.robots[3], self.m_objs[3], self.f_objs[2], self.f_objs[1]),
                'a18': ('transfer', self.robots[3], self.m_objs[3], self.f_objs[1], self.f_objs[2]),
                'a19': ('return', self.robots[3], None, None, None)
                }
        action_orders = {self.robots[0]: ('a0', 'a1', 'a2', 'a3', 'a4'),
                         self.robots[1]: ('a5', 'a6', 'a7', 'a8', 'a9'),
                         self.robots[2]: ('a10', 'a11', 'a12', 'a13', 'a14'),
                         self.robots[3]: ('a15', 'a16', 'a17', 'a18', 'a19')
                         }
        obj_orders = {self.m_objs[0]: ['a1', 'a11'],
                      self.m_objs[1]: ['a3', 'a13'],
                      self.m_objs[2]: ['a6', 'a16'],
                      self.m_objs[3]: ['a8', 'a18']
                      }
        init_order_constraints = ({'pre': 'a1', 'post': 'a10'}, {'pre': 'a3', 'post': 'a12'},
                                  {'pre': 'a6', 'post': 'a15'}, {'pre': 'a8', 'post': 'a17'})
        return plan, action_orders, obj_orders, init_order_constraints

    def _create_problem(self):
        other_arm = get_other_arm(self._arm)
        initial_conf = get_carry_conf(self._arm, self._grasp_type)

        plane = create_floor()
        room = load_pybullet(TWO_ROOMS)
        self.custom_limits = {}

        table = []
        table.append(load_pybullet(NARROW_TABLE))
        set_point(table[0], (0, -5, 0))
        table.append(load_pybullet(NARROW_TABLE))
        set_point(table[1], (0, 5, 0))
        shelf = create_shelf(w=SHELF_WIDTH, l=0.2, h=SHELF_HEIGHT,
                                  set_point=(0, 4.9, 0.6265), sim_id=self._sim_id)
        shelf_placement = create_shelf_placement(w=0.6265, l=0.14, h=0.01, color=BROWN)
        set_point(shelf_placement, (0, 4.9, 0.6265))
        middle_placement = create_shelf_placement(w=1.6265, l=0.2, h=0.01, color=RGBA(0.9297, 0.7930, 0.6758, 1))
        set_point(middle_placement, (0, 0, 0.49))
        self.fixed_obstacles = [shelf] + table

        boxes = []
        self.m_objs_init_placements = {}
        displacement_x = 0.4
        displacement_y = 0.05
        for i in range(self._num_m_objs):
            if i == 0: boxes.append(create_box(BOX[0], BOX[1], BOX[2], color=RED))
            elif i == 1: boxes.append(create_box(BOX[0], BOX[1], BOX[2], color=WHITE))
            elif i == 2: boxes.append(create_box(BOX[0], BOX[1], BOX[2], color=YELLOW))
            elif i == 3: boxes.append(create_box(BOX[0], BOX[1], BOX[2], color=GREEN))
            box_pose = ((-0.8 + displacement_x, -4.9 + displacement_y * pow(-1, i), 0.6265 + .15 / 2),
                        (0, 0, 0, 1))
            set_point(boxes[i], box_pose[0])
            set_quat(boxes[i], box_pose[1])
            self.m_objs_init_placements[boxes[i]] = Pose(boxes[i], box_pose, table[0])
            displacement_x += 0.3

        self.robots = {}
        self.robots_init_poses = {}
        displacement_x = 1.2
        displacement_y = 0.0
        rotate_z = 0.5
        for i in range(self._num_robots):
            self.robots[i] = create_pr2(robot=i)
            if i < 2:
                self.robots_init_poses[self.robots[i]] = (displacement_x * i, -2.5 + displacement_y * i, rotate_z * i)
                self.custom_limits[self.robots[i]] = {0: (-1.9, 1.9), 1: (-6., -0.1)}
            else:
                self.robots_init_poses[self.robots[i]] = (displacement_x * (i - 2), 2.5 + displacement_y * i, rotate_z * i)
                self.custom_limits[self.robots[i]] = {0: (-1.9, 1.9), 1: (0.1, 6.)}
            for j in range(len(joints_from_names(self.robots[i], PR2_GROUPS['base']))):
                set_joint_position(self.robots[i], joints_from_names(self.robots[i], PR2_GROUPS['base'])[j],
                                   self.robots_init_poses[self.robots[i]][j])
            set_arm_conf(self.robots[i], self._arm, initial_conf)
            open_arm(self.robots[i], self._arm)
            set_arm_conf(self.robots[i], other_arm, arm_conf(other_arm, REST_LEFT_ARM))
            close_arm(self.robots[i], other_arm)

        # comment out for visualization
        for robot in self.robots.values():
            self._grippers[robot] = get_gripper(robot)

        self.m_objs = boxes
        self.f_objs = [table[0], middle_placement, shelf_placement]
        self.m_obj_in_f_obj = {table[0]: set(boxes)}