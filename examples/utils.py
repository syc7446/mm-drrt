import os
import abc
import math
import time
import pickle
import numpy as np
import pybullet as p
import random
from itertools import islice, count
from datetime import datetime, date, time

from pybullet_utils.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from external.pybullet_planning.pybullet_tools.ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics
from external.pybullet_planning.pybullet_tools.pr2_primitives import create_trajectory, iterate_approach_path, \
    Commands, State, SELF_COLLISIONS, Pose, Conf
from external.pybullet_planning.pybullet_tools.pr2_utils import get_gripper_link, get_arm_joints, arm_conf, open_arm, \
    get_aabb, get_disabled_collisions, get_group_joints, learned_pose_generator, PR2_GROUPS
from external.pybullet_planning.pybullet_tools.utils import is_placement, multiply, invert, set_joint_positions, \
    pairwise_collision, get_joint_positions, plan_direct_joint_motion, plan_joint_motion, joint_from_name, all_between, \
    BodySaver, LockRenderer, get_bodies, get_joint_limits, set_joint_limits, get_default_resolution, uniform_pose_generator, \
    Saver, PoseSaver, ConfSaver, get_configuration, remove_body, inverse_kinematics_helper, get_movable_joints, \
    get_link_pose, is_pose_close, elapsed_time, irange, create_sub_robot, get_custom_limits, sub_inverse_kinematics, INF, \
    get_box_geometry, create_shape, create_body, sample_placement, get_pose, get_euler, STATIC_MASS, RED, BROWN, join_paths, \
    get_parent_dir, get_extend_fn, get_collision_fn, MAX_DISTANCE
from external.pybullet_planning.motion.motion_planners.utils import default_selector


MODEL_DIRECTORY = join_paths(get_parent_dir(__file__), os.pardir, 'examples/models/')
ROOM_FLOOR = join_paths(MODEL_DIRECTORY, 'room_floor.urdf')
SHORT_FLOOR = join_paths(MODEL_DIRECTORY, 'short_floor.urdf')
ROOMS = join_paths(MODEL_DIRECTORY, 'rooms.urdf')
SINGLE_ROOM = join_paths(MODEL_DIRECTORY, 'single_room.urdf')
SINGLE_BIG_ROOM = join_paths(MODEL_DIRECTORY, 'single_big_room.urdf')
SINGLE_SMALL_ROOM = join_paths(MODEL_DIRECTORY, 'single_small_room.urdf')
NARROW_TABLE = join_paths(MODEL_DIRECTORY, 'narrow_table.urdf')


# Table
TABLE_POSE_X = 0.0
TABLE_POSE_Y = 1.8

# Shelf
PACKING_CAMERA_POINT = (0, 2.5, 1.5)
PACKING_TARGET_POINT = (0, 0, 0)
SHELF_WIDTH = 0.6
SHELF_LENGTH = 0.5
SHELF_HEIGHT = 0.4
SHELF_REACHABLE_MARGIN = 1.0


def create_shelf(w, l, h, set_point, sim_id):
    link_vis = []
    link_cols = []
    link_pos = []

    # Left side
    link_cols.append(p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.01 / 2, l / 2, h / 2],
        physicsClientId=sim_id))
    link_vis.append(p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.01 / 2, l / 2, h / 2],
        rgbaColor=(0.6, 0.3, 0.0, 0.5),
        physicsClientId=sim_id))
    link_pos.append([set_point[0] - w / 2, set_point[1], set_point[2] + h / 2])
    # Right side
    link_cols.append(p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.01 / 2, l / 2, h / 2],
        physicsClientId=sim_id))
    link_vis.append(p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.01 / 2, l / 2, h / 2],
        rgbaColor=(0.6, 0.3, 0.0, 0.5),
        physicsClientId=sim_id))
    link_pos.append([set_point[0] + w / 2, set_point[1], set_point[2] + h / 2])
    # Back side
    link_cols.append(p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[w / 2, 0.01 / 2, h / 2],
        physicsClientId=sim_id))
    link_vis.append(p.createVisualShape(
        p.GEOM_BOX, halfExtents=[w / 2, 0.01 / 2, h / 2],
        rgbaColor=(0.6, 0.3, 0.0, 0.5),
        physicsClientId=sim_id))
    link_pos.append([set_point[0], set_point[1] + l / 2, set_point[2] + h / 2])
    # Top side
    link_cols.append(p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[w / 2, l / 2, 0.01 / 2],
        physicsClientId=sim_id))
    link_vis.append(p.createVisualShape(
        p.GEOM_BOX, halfExtents=[w / 2, l / 2, 0.01 / 2],
        rgbaColor=(0.6, 0.3, 0.0, 0.5),
        physicsClientId=sim_id))
    link_pos.append([set_point[0], set_point[1], set_point[2] + h])

    return p.createMultiBody(
        linkMasses=[10 for _ in link_pos],
        linkCollisionShapeIndices=link_cols,
        linkVisualShapeIndices=link_vis,
        linkPositions=link_pos,
        linkOrientations=[[0, 0, 0, 1] for _ in link_pos],
        linkInertialFramePositions=[[0, 0, 0] for _ in link_pos],
        linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in link_pos],
        linkParentIndices=[0 for _ in link_pos],
        linkJointTypes=[p.JOINT_FIXED for _ in link_pos],
        linkJointAxis=[[0, 0, 0] for _ in link_pos],
        physicsClientId=sim_id)


def create_shelf_placement(w, l, h, mass=STATIC_MASS, color=RED, **kwargs):
    collision_id, visual_id = create_shape(get_box_geometry(w, l, h), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)


class Environment:
    """Base class for an environment.
    """
    def __init__(self, num_objs, seed):
        self._num_objs = num_objs
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    @abc.abstractmethod
    def parse_state(self, state):
        """Parse the given state into literals.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_train_initial_states(self):
        """Returns a list of initial states for the training set.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def simulate(self, state, action):
        """Returns a next_state drawn from the transition model, along
        with a reward and done bit.
        """
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def literal_goal(self):
        """Get the goal expressed as a set of literals.
        """
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def discrete_predicates(self):
        """Get a set of all discrete (not continuous) predicates in this env.
        """
        raise NotImplementedError("Override me!")


class EnvironmentFailure(Exception):
    """Exception raised when something goes wrong in an environment.
    """
    pass