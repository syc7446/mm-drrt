#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import time
import copy
import random

from external.pybullet_planning.pybullet_tools.utils import get_pose, get_joint_positions, joints_from_names, is_placement, \
    set_base_values, load_pybullet, create_box, set_point, set_euler, set_joint_positions, TABLE_URDF, WorldSaver, \
    BROWN
from external.pybullet_planning.pybullet_tools.pr2_utils import get_other_arm, get_carry_conf, set_arm_conf, open_arm, \
    PR2_GROUPS, arm_conf, close_arm, REST_LEFT_ARM
from external.pybullet_planning.pybullet_tools.pr2_primitives import get_stable_gen, get_grasp_gen, Pose
from external.pybullet_planning.pybullet_tools.pr2_problems import create_pr2, create_floor, Problem, TABLE_MAX_Z

from examples.utils import SINGLE_ROOM, NARROW_TABLE, create_shelf, create_shelf_placement, TABLE_POSE_X, \
    TABLE_POSE_Y, SHELF_LENGTH, SHELF_WIDTH, SHELF_HEIGHT, SHELF_REACHABLE_MARGIN, Environment
from mm_drrt.utils.utils import get_ik_ir_gen, base_motion


class PickPlaceEnvironment(Environment):
    def __init__(self, num_objs, arm, grasp_type, sim_id, seed):
        self._num_objs = num_objs
        self._sim_id = sim_id
        self._objs = []
        for i in range(self._num_objs):
            self._objs.append("obj{}".format(i))
        self._objs_to_obj_ids = {}
        self._initialize(arm, grasp_type)

    def _initialize(self, arm, grasp_type):
        self._arm = arm
        self._problem = self._create_problem(grasp_type)
        self.grasp_gen_fn = get_grasp_gen(self._problem, collisions=True)
        self.placement_gen_fn = get_stable_gen(self._problem)
        for i, obj in enumerate(self._objs):
            self._objs_to_obj_ids[obj] = self._problem.movable[i]

    def pick_action(self, obj, pre_saved_world=None):
        movable_obstacles = copy.deepcopy(self._problem.movable)
        movable_obstacles.remove(self._objs_to_obj_ids[obj])
        collision_objs = self._problem.fixed + movable_obstacles
        ik_ir_fn = get_ik_ir_gen(self._problem,
                                 max_attempts=25, teleport=False,
                                 custom_limits=self.custom_limits, collision_objs=collision_objs)

        if not pre_saved_world:
            saved_world = WorldSaver()
        else:
            saved_world = pre_saved_world
            saved_world.restore()
        obj_pose = Pose(self._objs_to_obj_ids[obj])
        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))

        # Pick
        grasps = list(self.grasp_gen_fn(self._objs_to_obj_ids[obj]))
        (g,) = random.choice(grasps)

        pick_output = next(ik_ir_fn(self._arm, self._objs_to_obj_ids[obj], obj_pose, g), None)
        if not pick_output:
            print('Plan fails: pick in IsValidPickPlace')
            saved_world.restore()
            return g.value
        result_saved_world = WorldSaver()

        # Base motion for pick
        pick_base_path = base_motion(self.robot, base_start, pick_output[0].values, teleport=False,
                                     obstacles=self._problem.fixed, custom_limits=self.custom_limits)
        if not pick_base_path:
            print('Plan fails: base pick motion in IsValidPickPlace')
            saved_world.restore()
            return g.value
        set_joint_positions(self.robot, [0, 1, 2], pick_base_path[-1])

        basex, basey, basez = pick_output[0].values
        gripx, gripy, gripz = pick_output[3]
        return {'saved_world': result_saved_world, 'config': [None, g, pick_output[0]],
                0: basex, 1: basey, 2: basez, 3: gripx, 4: gripy, 5: gripz}

    def place_action(self, obj, grasp, pre_saved_world=None):
        movable_obstacles = copy.deepcopy(self._problem.movable)
        movable_obstacles.remove(self._objs_to_obj_ids[obj])
        collision_objs = self._problem.fixed + movable_obstacles
        ik_ir_fn = get_ik_ir_gen(self._problem,
                                 max_attempts=25, teleport=False,
                                 custom_limits=self.custom_limits, collision_objs=collision_objs)

        if not pre_saved_world:
            saved_world = WorldSaver()
        else:
            saved_world = pre_saved_world
            saved_world.restore()

        # Place
        placement_gen = self.placement_gen_fn(self._objs_to_obj_ids[obj], self.shelf_placement)
        (p,) = next(placement_gen)

        attachment = grasp.get_attachment(self.robot, self._arm)
        place_output = next(ik_ir_fn(self._arm, self._objs_to_obj_ids[obj], p, grasp), None)
        if not place_output:
            print('Plan fails: place in IsValidPickPlace')
            saved_world.restore()
            return grasp.value
        result_saved_world = WorldSaver()

        # Base motion for place
        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        place_base_path = base_motion(self.robot, base_start, place_output[0].values,
                                      teleport=False, obstacles=self._problem.fixed,
                                      attachments=[attachment], custom_limits=self.custom_limits)
        if not place_base_path:
            print('Plan fails: base motion in IsValidPickPlace')
            saved_world.restore()
            return p.value
        set_joint_positions(self.robot, [0, 1, 2], place_base_path[-1])

        result_saved_world.restore()
        basex, basey, basez = place_output[0].values
        gripx, gripy, gripz = place_output[3]
        return {'saved_world': result_saved_world, 'config': [p, None, place_output[0]],
                0: basex, 1: basey, 2: basez, 3: gripx, 4: gripy, 5: gripz}

    def _create_problem(self, grasp_type):
        other_arm = get_other_arm(self._arm)
        initial_conf = get_carry_conf(self._arm, grasp_type)

        plane = create_floor()
        room = load_pybullet(SINGLE_ROOM)
        self.custom_limits = {0: (-3., 3.), 1: (-3., 3.)}

        self.table = []
        self.table.append(load_pybullet(NARROW_TABLE))
        set_point(self.table[0], (-1.2, -1.4, 0))
        self.table.append(load_pybullet(NARROW_TABLE))
        set_point(self.table[1], (1.2, -1.4, 0))
        self.table.append(load_pybullet(TABLE_URDF))
        set_point(self.table[2], (0, 2, 0))
        self.shelf = create_shelf(w=SHELF_WIDTH, l=SHELF_LENGTH, h=SHELF_HEIGHT,
                                  set_point=(TABLE_POSE_X, TABLE_POSE_Y, TABLE_MAX_Z), sim_id=self._sim_id)
        self.shelf_placement = create_shelf_placement(w=SHELF_WIDTH, l=SHELF_LENGTH, h=0.01, color=BROWN)
        set_point(self.shelf_placement, (TABLE_POSE_X, TABLE_POSE_Y, TABLE_MAX_Z))
        self.reachable_point = (-SHELF_REACHABLE_MARGIN, 0.0, 0.0)

        boxes = []
        displacement_x = 0
        displacement_y = 0.1
        for i in range(self._num_objs):
            boxes.append(create_box(.07, .05, .15))
            set_point(boxes[i], (-1.8 + displacement_x, -1.4 + displacement_y * pow(-1, i), TABLE_MAX_Z + .15 / 2))
            set_euler(boxes[i], (0, 0, 0))
            displacement_x += 0.3

        self.robot = create_pr2()
        set_base_values(self.robot, (0, 0, 0))
        set_arm_conf(self.robot, self._arm, initial_conf)
        open_arm(self.robot, self._arm)
        set_arm_conf(self.robot, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
        close_arm(self.robot, other_arm)

        # bottom_aabb = get_aabb(bottom_body, link=bottom_link)

        return Problem(robot=self.robot, movable=boxes, arms=[self._arm], grasp_types=[grasp_type],
                       surfaces=[self.table[0], self.shelf_placement])