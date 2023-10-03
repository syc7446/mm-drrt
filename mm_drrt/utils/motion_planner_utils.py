import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import random
from random import choice
import math
import copy
import time

from external.pybullet_planning.pybullet_tools.ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics
from external.pybullet_planning.pybullet_tools.pr2_utils import get_gripper_link, get_arm_joints, arm_conf, open_arm, \
    get_aabb, get_disabled_collisions, get_group_joints, learned_pose_generator, create_gripper, PR2_GROUPS, GET_GRASPS, \
    get_top_grasps, get_side_grasps, compute_grasp_width, get_torso_arm_joints, TOP_HOLDING_LEFT_ARM, SIDE_HOLDING_LEFT_ARM
from external.pybullet_planning.pybullet_tools.pr2_primitives import create_trajectory, iterate_approach_path, Commands, \
    State, SELF_COLLISIONS, Pose, Conf, Grasp
from external.pybullet_planning.pybullet_tools.utils import is_placement, multiply, invert, set_joint_positions, \
    pairwise_collision, get_sample_fn, get_distance_fn, check_initial_end, \
    get_joint_positions, plan_direct_joint_motion, joint_from_name, all_between, BodySaver, \
    LockRenderer, get_bodies, get_joint_limits, set_joint_limits, get_default_resolution, uniform_pose_generator, Saver, \
    PoseSaver, ConfSaver, remove_body, inverse_kinematics_helper, get_movable_joints, get_link_pose, \
    is_pose_close, elapsed_time, irange, create_sub_robot, get_custom_limits, sub_inverse_kinematics, INF, \
    get_box_geometry, create_shape, create_body, sample_placement, get_pose, get_euler, STATIC_MASS, RED, BROWN, \
    join_paths, get_parent_dir, get_extend_fn, get_collision_fn, unit_quat, get_unit_vector, MAX_DISTANCE, \
    remove_redundant, all_close
from external.pybullet_planning.motion.motion_planners.primitives import distance_fn_from_extend_fn
from external.pybullet_planning.motion.motion_planners.utils import get_pairs, default_selector, get_distance, \
    is_path, flatten

from mm_drrt.planner.prm import prm


GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH


def plan_joint_motion(robot, joints, end_conf, obstacles=[], attachments=[],
                      self_collisions=True, disabled_collisions=set(),
                      weights=None, resolutions=None, max_distance=MAX_DISTANCE,
                      use_aabb=False, cache=True, custom_limits={}, use_drrt_star=False,
                      use_debug_plot=False, **kwargs):

    assert len(joints) == len(end_conf)
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(robot, joints, weights=weights)
    extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(robot, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance,
                                    use_aabb=use_aabb, cache=cache)

    start_conf = get_joint_positions(robot, joints)
    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None
    roadmap, heuristic_val = prm(start_conf, end_conf, distance_fn, sample_fn, extend_fn,
                                 collision_fn, use_drrt_star=use_drrt_star,
                                 use_debug_plot=use_debug_plot, **kwargs)
    return roadmap, heuristic_val


def base_motion(robot, base_start, base_goal, teleport=False, obstacles=[], attachments=[], custom_limits={},
                num_samples=100, expand_type=None, expand_configs=None, use_debug=False):
    disabled_collisions = get_disabled_collisions(robot)
    base_joints = [joint_from_name(robot, name) for name in PR2_GROUPS['base']]
    set_joint_positions(robot, base_joints, base_start)
    base_goal = base_goal[:len(base_joints)]
    # TODO: hardcoded values to increase resolutions used in extension fn

    if teleport:
        set_joint_positions(robot, base_joints, base_start)
        if any(pairwise_collision(robot, b) for b in obstacles):
            return None
        set_joint_positions(robot, base_joints, base_goal)
        if any(pairwise_collision(robot, b) for b in obstacles):
            return None
        return [base_start, base_goal]
    else:
        resolutions = np.array([2 * get_default_resolution(robot, 2), 2 * get_default_resolution(robot, 2),
                                get_default_resolution(robot, 2)])
        with LockRenderer(lock=False):
            for _ in range(5):
                roadmap, heuristic_val = sub_plan_joint_motion(robot, base_joints, base_goal, obstacles=obstacles,
                                                               attachments=attachments, disabled_collisions=disabled_collisions,
                                                               resolutions=resolutions, custom_limits=custom_limits,
                                                               use_drrt_star=True,
                                                               num_samples=num_samples, expand_type=expand_type, expand_configs=expand_configs,
                                                               use_debug=use_debug)
                if roadmap == None:
                    num_samples += 20
                    set_joint_positions(robot, base_joints, base_start)
                else:
                    break
        if not roadmap: set_joint_positions(robot, base_joints, base_start)
        set_joint_positions(robot, base_joints, base_goal)
        return roadmap, heuristic_val


def arm_retrieval_motion(robot, arm, type, grasp, start, goal, obstacles=[], attachments=[], custom_limits={},
                         num_samples=100, expand_type=None, expand_configs=None, use_debug=False):
    disabled_collisions = get_disabled_collisions(robot)
    arm_joints = get_arm_joints(robot, arm)
    set_joint_positions(robot, arm_joints, start)
    resolutions = 0.05 ** np.ones(len(arm_joints))
    with LockRenderer(lock=False):
        for _ in range(5):
            roadmap, heuristic_val = sub_plan_joint_motion(robot, arm_joints, goal, obstacles=obstacles,
                                                           attachments=attachments, disabled_collisions=disabled_collisions,
                                                           self_collisions=False,
                                                           resolutions=resolutions, custom_limits=custom_limits,
                                                           use_drrt_star=True, num_samples=num_samples,
                                                           expand_type=expand_type, expand_configs=expand_configs,
                                                           use_debug=use_debug)
            if roadmap == None:
                num_samples += 20
                set_joint_positions(robot, arm_joints, start)
            else:
                break
    if not roadmap:
        set_joint_positions(robot, arm_joints, start)
        return None, None
    set_joint_positions(robot, arm_joints, goal)
    return roadmap, heuristic_val


def get_ir_sampler(robot, gripper, custom_limits={}, max_attempts=25, collisions=True, collision_objs=[], num_samples=100, learned=True, use_debug=False):
    robot = robot
    obstacles = collision_objs if collisions else []
    gripper = gripper

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                return
        gripper_pose = multiply(pose.value, invert(grasp.value))  # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = arm_conf(arm, grasp.carry)
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_group_joints(robot, 'base')
        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)
        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                # print('IR attempts:', count)
                yield (bq,)
                break
            else:
                yield None

    return gen_fn


def get_grasp_gen(grasp_type, collisions=False, randomize=True):
    if grasp_type not in GET_GRASPS:
        raise ValueError('Unexpected grasp type:', grasp_type)
    def fn(robot, body):
        # TODO: max_grasps
        # TODO: return grasps one by one
        grasps = []
        arm = 'left'
        #carry_conf = get_carry_conf(arm, 'top')
        if 'top' == grasp_type:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([1, 0, 0])
            grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
                          for g in get_top_grasps(body, grasp_length=GRASP_LENGTH))
        if 'side' == grasp_type:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([2, 0, -1])
            grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
                          for g in get_side_grasps(body, grasp_length=GRASP_LENGTH))
        filtered_grasps = []
        for grasp in grasps:
            grasp_width = compute_grasp_width(robot, arm, body, grasp.value) if collisions else 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        if randomize:
            random.shuffle(filtered_grasps)
        return [(g,) for g in filtered_grasps]
        #for g in filtered_grasps:
        #    yield (g,)
    return fn


def get_placement_gen(**kwargs):
    def gen(body, surface):
        while True:
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            yield (p,)
    return gen


def get_stable_gen(collisions=True, **kwargs):
    def gen(body, surface, obstacles):
        while True:
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    # TODO: apply the acceleration technique here
    return gen


def get_ik_fn(robot, custom_limits={}, collisions=True, collision_objs=[], use_debug=False):
    robot = robot
    obstacles = collision_objs if collisions else []
    if use_debug:
        if is_ik_compiled():
            print('Using ikfast for inverse kinematics')
        else:
            print('Using pybullet for inverse kinematics')

    def fn(arm, obj, pose, grasp, base_conf):
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value))  # w_f_g = w_f_o * (g_f_o)^-1
        # approach_pose = multiply(grasp.approach, gripper_pose)
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_arm_joints(robot, arm)

        default_conf = arm_conf(arm, grasp.carry)
        # sample_fn = get_sample_fn(robot, arm_joints)
        pose.assign()
        base_conf.assign()
        open_arm(robot, arm)
        set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()
        grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits)  # , upper_limits=USE_CURRENT)
        # nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):  # [obj]
            # print('Grasp IK failure', grasp_conf)
            if use_debug:
                if grasp_conf is not None:
                   print('grasp_conf', grasp_conf)
                #    #wait_if_gui()
                if any(pairwise_collision(robot, b) for b in obstacles):
                    print('Collision exists', obstacles)
            return (None, None,)
        # approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
        #                                       upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose,
                                               custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            if use_debug:
                print('Approach IK failure', approach_conf)
            # wait_if_gui()
            return (None, None,)
        approach_conf = get_joint_positions(robot, arm_joints)
        return (approach_conf, grasp_conf,)

    return fn


def get_arm_motion_fn(robot, custom_limits={}, collisions=True, collision_objs=[], num_samples=100,
                      expand_type=None, expand_configs=None, use_debug=False):
    robot = robot
    obstacles = collision_objs if collisions else []
    if use_debug:
        if is_ik_compiled():
            print('Using ikfast for inverse kinematics')
        else:
            print('Using pybullet for inverse kinematics')

    def fn(arm, obj, grasp, approach_conf, grasp_conf, attachments=[]):
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        arm_joints = get_arm_joints(robot, arm)

        default_conf = arm_conf(arm, grasp.carry)
        roadmap, heuristic_val = None, None

        resolutions = 0.05 ** np.ones(len(arm_joints))
        set_joint_positions(robot, arm_joints, default_conf)
        _num_samples = num_samples
        for _ in range(5):
            roadmap, heuristic_val = sub_plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments,
                                                           obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                                           custom_limits=custom_limits, resolutions=resolutions,
                                                           use_drrt_star=True,
                                                           num_samples=_num_samples, expand_type=expand_type, expand_configs=expand_configs,
                                                           use_debug=use_debug)
            if roadmap == None:
                _num_samples += 20
                set_joint_positions(robot, arm_joints, default_conf)
            else:
                break
        if roadmap is None:
            if use_debug:
                print('Roadmap is not found')
            set_joint_positions(robot, arm_joints, default_conf)
            return (None, None,)
        approach_path = roadmap(expand_configs + default_conf, expand_configs + approach_conf)
        # approach_path = roadmap(default_conf, approach_conf)
        if approach_path is None:
            if use_debug:
                print('Approach path failure')
            return (None, None,)
        set_joint_positions(robot, arm_joints, approach_conf)
        grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments,
                                              obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions / 2.)
        if grasp_path is None:
            if use_debug:
                print('Grasp path failure')
            return (None, None,)
        set_joint_positions(robot, arm_joints, grasp_conf)

        grasp_conf = expand_configs + grasp_conf
        grasp_path = [expand_configs + g for g in grasp_path]
        roadmap.add([grasp_conf])
        roadmap.connect(roadmap.vertices[roadmap.final_conf], roadmap.vertices[grasp_conf], grasp_path)
        roadmap.final_conf = grasp_conf
        for v in heuristic_val.keys(): heuristic_val[v] += 1.0
        heuristic_val[roadmap.vertices[grasp_conf]] = 0.0

        #     path = approach_path + grasp_path
        # mt = create_trajectory(robot, arm_joints, path)
        # cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        # return (cmd, attachments, gripper_pose[0], roadmap, )  # Only this line has been changed from the original code
        return (roadmap, heuristic_val,)

    return fn


def get_ik_ir_gen(robot, gripper, max_attempts=25, learned=True, use_debug=False, **kwargs):
    # TODO: compose using general fn
    ir_sampler = get_ir_sampler(robot, gripper, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_fn(robot, **kwargs)

    def gen(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if use_debug:
                print('attempts', attempts)
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            if use_debug:
                print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            # if not p.init:
            #    return

    return gen


def get_gripper(robot_id, arm='left', visual=True):
    # upper = get_max_limit(problem.robot, get_gripper_joints(problem.robot, 'left')[0])
    # set_configuration(gripper, [0]*4)
    # dump_body(gripper)
    return create_gripper(robot_id, arm=arm, visual=visual)


def get_configuration(robot, name):
    return get_joint_positions(robot, get_group_joints(robot, name))


def get_arm_positions(robot, arm):
    arm_joints = get_arm_joints(robot, arm)
    arm_positions = get_joint_positions(robot, arm_joints)
    return arm_positions


class Problem(object):
    def __init__(self, robots, arms=tuple(), movable=tuple(), grasp_types=tuple(),
                 init_placement=tuple(), surfaces=tuple(), sinks=tuple(), stoves=tuple(),
                 buttons=tuple(), goal_conf=None, goal_holding=tuple(), goal_on=tuple(),
                 goal_cleaned=tuple(), goal_cooked=tuple(), costs=False,
                 body_names={}, body_types=[], base_limits=None):
        self.robots = robots
        self.arms = arms
        self.movable = movable
        self.collision_objs = list(filter(lambda b: b not in [robots[r] for r in robots], get_bodies()))
        self.grasp_types = grasp_types
        self.init_placement = init_placement
        self.surfaces = surfaces
        self.sinks = sinks
        self.stoves = stoves
        self.buttons = buttons
        self.goal_conf = goal_conf
        self.goal_holding = goal_holding
        self.goal_on = goal_on
        self.goal_cleaned = goal_cleaned
        self.goal_cooked = goal_cooked
        self.costs = costs
        self.body_names = body_names
        self.body_types = body_types
        self.base_limits = base_limits
        self.gripper = None
    def get_gripper(self, robot_id, arm='left', visual=True):
        # upper = get_max_limit(problem.robot, get_gripper_joints(problem.robot, 'left')[0])
        # set_configuration(gripper, [0]*4)
        # dump_body(gripper)
        if self.gripper is None:
            self.gripper = create_gripper(robot_id, arm=arm, visual=visual)
        return self.gripper
    def remove_gripper(self):
        if self.gripper is not None:
            remove_body(self.gripper)
            self.gripper = None
    def __repr__(self):
        return repr(self.__dict__)


# dRRT

class TreeNode(object):

    def __init__(self, config, num_robots=1, parent=None, path=[]):
        self.config = config
        self.num_robots = num_robots
        self.parent = parent

        self.sub_config = []
        for r in range(self.num_robots):
            self.sub_config.append(self.config[r*int(len(config)/num_robots):
                                               (r+1)*int(len(config)/num_robots)])
        self.sub_local_paths = path

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def __str__(self):
        return 'TreeNode(' + str(self.config) + ')'
    __repr__ = __str__


def get_sub_nodes(nodes, robot_id):
    sub_nodes = []
    for i in range(len(nodes)):
        sub_nodes.append(nodes[i].sub_config[robot_id])
    return sub_nodes


def get_sub_q_near(nodes, roadmaps, sub_samples, subprob_id):
    sub_q_near = []
    for r in range(len(roadmaps)):
        sub_q_near.append(nodes[get_max_dist_node_index(nodes, roadmaps, sub_samples, subprob_id)].sub_config[r])
    return sub_q_near


def get_max_length_list(list):
    list_len = [len(i) for i in list]
    if list_len: return max(list_len)
    else: return 0


def get_local_paths(roadmap=[], num_robots=1, start_configs=[], target_configs=[]):
    local_paths = []
    for r in range(num_robots):
        local_path = list(roadmap[r](start_configs[r], target_configs[r]))[:-1]
        if not local_path: local_path = [start_configs[r]]
        local_paths.append(local_path)
    return local_paths


def is_local_collision(paths, drrt_collision_fn=None):
    for i in range(max([len(paths[r]) for r in range(len(paths))])):
        configs = []
        for r in range(len(paths)):
            if i >= len(paths[r]): configs.append(paths[r][-1])
            else: configs.append(paths[r][i])
        if drrt_collision_fn(configs, mode='boolean'):
            return True
    return False


def search(list, element):
    for i in range(len(list)):
        if list[i] == element:
            return True
    return False


def get_collision_free_paths_to_target(paths, drrt_collision_fn=None):
    collision_robot_index = []
    for i in range(max([len(paths[r]) for r in range(len(paths))])):
        if len(collision_robot_index) == len(paths): break
        configs = []
        for r in range(len(paths)):
            if i >= len(paths[r]): configs.append(paths[r][-1])
            else: configs.append(paths[r][i])
        skip_collision_robot_index = []
        for r in range(len(paths)):
            if search(collision_robot_index, r): skip_collision_robot_index.append(r)
        r_collision_robot_index =  drrt_collision_fn(configs, mode='index', skip_index=skip_collision_robot_index)
        collision_robot_index += r_collision_robot_index
        collision_robot_index = list(np.unique(collision_robot_index))
    if not collision_robot_index: return paths
    else:
        path_lengths = [len(paths[r]) for r in range(len(paths))]
        accu_path_length = 0
        for r in range(len(collision_robot_index)):
            if r == 0: continue
            accu_path_length += path_lengths[r - 1]
            paths[collision_robot_index[r]] = [paths[collision_robot_index[r]][0] for _ in range(accu_path_length)] \
                                              + paths[collision_robot_index[r]]
        return paths


# def get_inter_robots_collision_fn(robots, joints, attachments=[], robot_id=0, num_robots=1,
#                                   use_aabb=False, cache=False, max_distance=MAX_DISTANCE, **kwargs):
#     moving_links = frozenset(link for link in get_moving_links(robots[robot_id], joints[robot_id])
#                              if can_collide(robots[robot_id], link)) # TODO: propagate elsewhere
#     attached_bodies = [attachment.child for attachment in attachments]
#     moving_bodies = [CollisionPair(robots[robot_id], moving_links)] + list(map(parse_body, attached_bodies))
#     get_obstacle_aabb = cached_fn(get_buffered_aabb, cache=cache, max_distance=max_distance/2., **kwargs)
#
#     def drrt_collision_fn(q, verbose=False):
#         obstacles = []
#         for r in range(num_robots):
#             set_joint_positions(robots[r], joints[r], q[r])
#             if r == robot_id: # TODO: implement for obstacle robots
#                 for attachment in attachments:
#                     attachment.assign()
#             else:
#                 obstacles.append(robots[r])
#         #wait_for_duration(1e-2)
#         get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance/2., **kwargs)
#
#         for body1, body2 in product(moving_bodies, obstacles):
#             if (not use_aabb or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))) \
#                     and pairwise_collision(body1, body2, **kwargs):
#                 if verbose: print(body1, body2)
#                 return True
#         return False
#     return drrt_collision_fn


def get_inter_robots_collision_fn(robots, joints, num_robots=1, **kwargs):
    def drrt_collision_fn(q, mode='boolean', skip_index=[]):
        for r in range(num_robots):
            set_joint_positions(robots[r], joints[r], q[r])
            # TODO: attached objects are not currently considered for collision checking; only robot bodies are checking
            # for attachment in attachments:
            #     attachment.assign()

        if mode == 'boolean':
            for r in range(num_robots - 1):
                for r_prime in range(r + 1, num_robots):
                    if pairwise_collision(robots[r], robots[r_prime], **kwargs):
                        return True
            return False
        elif mode == 'index':
            collision_robot_index = []
            for r in range(num_robots - 1):
                if r in skip_index: continue
                for r_prime in range(r + 1, num_robots):
                    if r_prime in skip_index: continue
                    if pairwise_collision(robots[r], robots[r_prime], **kwargs):
                        collision_robot_index.append(r)
                        collision_robot_index.append(r_prime)
            return list(np.unique(collision_robot_index))
    return drrt_collision_fn


def sub_plan_joint_motion(robot, joints, end_conf, obstacles=[], attachments=[],
                          self_collisions=True, disabled_collisions=set(),
                          weights=None, resolutions=None, max_distance=MAX_DISTANCE,
                          use_aabb=False, cache=True, custom_limits={}, use_drrt_star=False,
                          use_debug_plot=False, **kwargs):

    assert len(joints) == len(end_conf)
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sub_sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    sub_distance_fn = get_distance_fn(robot, joints, weights=weights)
    sub_extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    sub_collision_fn = get_collision_fn(robot, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                        custom_limits=custom_limits, max_distance=max_distance,
                                        use_aabb=use_aabb, cache=cache)

    start_conf = get_joint_positions(robot, joints)
    if not check_initial_end(start_conf, end_conf, sub_collision_fn):
        return None, None
    roadmap, heuristic_val = prm(start_conf, end_conf, sub_distance_fn, sub_sample_fn, sub_extend_fn,
                                 sub_collision_fn, attachments=attachments, use_drrt_star=use_drrt_star,
                                 use_debug_plot=use_debug_plot, debug_roadmap_fn=debug_2d_sub_roadmap, **kwargs)
    return roadmap, heuristic_val


def get_max_dist_node_index(nodes, roadmaps, goal, subprob_id):
    dist = []
    for n in nodes:
        if n.subprob_id != subprob_id:
            dist.append(INF)
            continue
        sub_dist = 0
        for r in range(len(roadmaps)):
            sub_dist += roadmaps[r].sub_distance_fn(n.sub_config[r], goal[r])
        dist.append(sub_dist)
    return np.argmin(dist)


def connect_to_target(roadmap=[], num_robots=1, start_configs=[], target_configs=[], drrt_collision_fn=None):
    # inter-robot collisions for starts and goals
    if drrt_collision_fn(start_configs): return []
    if drrt_collision_fn(target_configs): return []
    for r in range(num_robots):
        configs = []
        for c in range(len(target_configs)):
            if c == r: configs.append(start_configs[c])
            configs.append(target_configs[c])
        if drrt_collision_fn(configs, mode='boolean'): return []
        configs = []
        for c in range(len(start_configs)):
            if c == r: configs.append(target_configs[c])
            configs.append(start_configs[c])
        if drrt_collision_fn(configs, mode='boolean'): return []

    local_paths = []
    for r in range(num_robots):
        if roadmap[r].expand_type == 'base':
            s_configs = start_configs[r][len(roadmap[r].expand_configs):]
            t_configs = target_configs[r][len(roadmap[r].expand_configs):]
        elif roadmap[r].expand_type == 'arm':
            s_configs = start_configs[r][:-1 * len(roadmap[r].expand_configs)]
            t_configs = target_configs[r][:-1 * len(roadmap[r].expand_configs)]
        local_path = list(roadmap[r].sub_extend_fn(s_configs, t_configs))[:-1]
        if not local_path: local_path = [start_configs[r]]
        if any(roadmap[r].sub_collision_fn(q) for q in local_path):
            return []
        if roadmap[r].expand_type == 'base':
            local_path = [roadmap[r].expand_configs + p for p in local_path]
        elif roadmap[r].expand_type == 'arm':
            local_path = [p + roadmap[r].expand_configs for p in local_path]
        local_paths.append(local_path)
    return local_paths


def get_sub_samples(roadmap, sub_samples):
    if roadmap.expand_type == 'base':
        sub_samples.append(roadmap.expand_configs + roadmap.sub_sample_fn())
    elif roadmap.expand_type == 'arm':
        sub_samples.append(roadmap.sub_sample_fn() + roadmap.expand_configs)
    return sub_samples


def get_angle(a, b, c):
    ang = abs(math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])))
    return 360 - ang if ang > 180 else ang


def is_duplicate(list_elems, new_elem, subprob_id):
    for elem in list_elems:
        if elem.subprob_id == subprob_id:
            if elem.config == new_elem:
                return True
    return False


def get_parent_node_index(nodes, q_near):
    near_node = ()
    for q in q_near:
        near_node += q
    for i in range(len(nodes)):
        if nodes[i].config == near_node: return i


def set_random_seed(seed=None):
    if seed is not None:
        random.seed(seed)


def update_subprob_id(sub_q, sub_goals, subprob_id, goals, joint_dim, roadmaps):
    sub_q_last = get_sub_q_list(sub_q, len(roadmaps))
    sub_q = get_sub_q_list(sub_q, len(roadmaps))
    for r in range(len(roadmaps)):
        if sub_q[r] == sub_goals[r]:
            if len(roadmaps[r]) - 1 > subprob_id[r]:
                subprob_id[r] += 1
                sub_goals[r] = get_sub_q_list(get_substarts_subgoals(goals, subprob_id, joint_dim), len(roadmaps))[r]
    return sub_q_last, sub_goals


##### path smoothing

def get_smoothing_fn(robot, joints, obstacles=[], attachments=[],
                          self_collisions=True, disabled_collisions=set(),
                          weights=None, resolutions=None, max_distance=MAX_DISTANCE,
                          use_aabb=False, cache=True, custom_limits={}):
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sub_sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    sub_distance_fn = get_distance_fn(robot, joints, weights=weights)
    sub_extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    sub_collision_fn = get_collision_fn(robot, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                        custom_limits=custom_limits, max_distance=max_distance,
                                        use_aabb=use_aabb, cache=cache)
    return sub_sample_fn, sub_distance_fn, sub_extend_fn, sub_collision_fn


def smooth_path(path, robots, sub_extend_fns, sub_collision_fns, sub_distance_fns=None, cost_fn=None, sub_sample_fns=None,
                max_iterations=50, max_time=INF, converge_time=INF, verbose=False):
    """
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_iterations: Maximum number of iterations - int
    :param max_time: Maximum runtime - float
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: makes an assumption on extend_fn (to avoid sampling the same segment)
    # TODO: smooth until convergence
    # TODO: dynamic expansion of the nearby graph
    start_time = last_time = time.time()
    if (path is None) or (max_iterations is None):
        return path
    assert (max_iterations < INF) or (max_time < INF)
    if cost_fn is None:
        sub_cost_fns = sub_distance_fns
    # TODO: use extend_fn to extract waypoints
    waypoints, waypoints_paths = waypoints_from_path(path, difference_fn=None) # TODO: difference_fn
    cost = compute_path_cost(waypoints, sub_cost_fns=sub_cost_fns)
    len_q_each = int(len(path[0]) / len(sub_cost_fns))
    #paths = [extend_fn(*pair) for pair in get_pairs(waypoints)] # TODO: update incrementally
    #costs = [cost_fn(*pair) for pair in get_pairs(waypoints)]
    for iteration in irange(max_iterations):
        if (elapsed_time(start_time) > max_time) or (elapsed_time(last_time) > converge_time) or (len(waypoints) <= 2):
            break

        segments = get_pairs(range(len(waypoints)))
        weights = [sum(sub_distance_fns[r](waypoints[i][len_q_each * r : len_q_each * (r + 1)],
                                       waypoints[j][len_q_each * r : len_q_each * (r + 1)])
                   for r in range(len(sub_distance_fns))) for i, j in segments]
        # paths = get_extend_path(sub_extend_fns, waypoints)
        #weights = [len(paths[i]) for i, j in segments]
        probabilities = np.array(weights) / sum(weights)
        if verbose:
            print('Iteration: {} | Waypoints: {} | Cost: {:.3f} | Elapsed: {:.3f} | Remaining: {:.3f}'.format(
                iteration, len(waypoints), cost, elapsed_time(start_time), max_time-elapsed_time(start_time)))

        #segment1, segment2 = choices(segments, weights=probabilities, k=2)
        seg_indices = list(range(len(segments)))
        seg_idx1, seg_idx2 = np.random.choice(seg_indices, size=2, replace=True, p=probabilities)
        if seg_idx1 == seg_idx2: # TODO: ensure not too far away
            continue
        if seg_idx2 < seg_idx1: # choices samples with replacement
            seg_idx1, seg_idx2 = seg_idx2, seg_idx1
        segment1, segment2 = segments[seg_idx1], segments[seg_idx2]
        # TODO: option to sample_fn only adjacent pairs
        #point1, point2 = [convex_combination(waypoints[i], waypoints[j], w=random())
        #                  for i, j in [segment1, segment2]]
        # point1, point2 = [waypoints_paths[i] for i, j in [segment1, segment2]]

        i, _ = segment1
        _, j = segment2
        point1, point2 = waypoints[i], waypoints[j]
        shortcut = [point1, point2]
        # if sample_fn is not None:
        #     shortcut = [point1, sample_fn(), point2]
        #shortcut_paths = [extend_fn(*pair) for pair in get_pairs(waypoints)]
        # new_waypoints = waypoints[:i+1] + shortcut + waypoints[j:] # TODO: reuse computation
        new_waypoints = waypoints[:i] + shortcut + waypoints[j + 1:]
        new_cost = compute_path_cost(new_waypoints, sub_cost_fns=sub_cost_fns)
        if new_cost >= cost: # TODO: cost must have percent improvement above a threshold
            continue
        if not any(smooth_collision_check(q, robots, sub_collision_fns) for q in default_selector(refine_waypoints(shortcut, sub_extend_fns))):
            waypoints = new_waypoints
            is_init_q_removed = False
            if i == 0:
                is_init_q_removed = True
            del waypoints_paths[i : j]
            if not is_init_q_removed:
                waypoints_paths.insert(i, refine_waypoints(shortcut, sub_extend_fns))
            else:
                waypoints_paths.insert(i, [shortcut[0]] + refine_waypoints(shortcut, sub_extend_fns))
            cost = new_cost
            last_time = time.time()
    #return waypoints
    refined_waypoints = []
    for waypoints_path in waypoints_paths:
        refined_waypoints += waypoints_path
    return refined_waypoints


def waypoints_from_path(path, difference_fn=None, tolerance=1e-3):
    if difference_fn is None:
        difference_fn = lambda q2, q1: np.array(q2) - np.array(q1) # get_difference
        #difference_fn = get_difference_fn(body, joints) # TODO: account for wrap around or use adjust_path
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path

    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    waypoints_paths = []
    waypoints_path = [path[0], path[1]]
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not all_close(last_difference, difference, atol=tolerance):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
            waypoints_paths.append(waypoints_path)
            waypoints_path = []
        last_conf = conf
        last_difference = difference
        waypoints_path.append(conf)
    waypoints.append(last_conf)
    waypoints_paths.append(waypoints_path)
    return waypoints, waypoints_paths


def compute_path_cost(path, sub_cost_fns=get_distance):
    if not is_path(path):
        return INF
    #path = waypoints_from_path(path)
    len_q_each = int(len(path[0]) / len(sub_cost_fns))
    cost_sum = 0
    for r in range(len(sub_cost_fns)):
        sub_path = [p[len_q_each * r : len_q_each * (r + 1)] for p in path]
        cost_sum += sum(sub_cost_fns[r](*pair) for pair in get_pairs(sub_path))
    return cost_sum


def get_extend_path(sub_extend_fns, waypoints):
    len_q_each = int(len(waypoints[0]) / len(sub_extend_fns))
    sub_waypoints, sub_pairs = [], []
    for r in range(len(sub_extend_fns)):
        sub_waypoints.append([p[len_q_each * r : len_q_each * (r + 1)] for p in waypoints])
        sub_pairs.append(get_pairs(sub_waypoints[r]))
    paths = []
    for i in range(len(sub_pairs[0])):
        sub_paths = []
        for r in range(len(sub_extend_fns)):
            sub_paths.append(list(sub_extend_fns[r](*sub_pairs[r][i])))
        merged_sub_paths = []
        for j in range(get_max_length_list(sub_paths)):
            q = ()
            for r in range(len(sub_extend_fns)):
                if len(sub_paths[r]) <= j:
                    q += sub_paths[r][len(sub_paths[r]) - 1]
                else:
                    q += sub_paths[r][j]
            merged_sub_paths.append(q)
        paths.append(merged_sub_paths)
    return paths


def refine_waypoints(waypoints, sub_extend_fns):
    #if len(waypoints) <= 1:
    #    return waypoints
    len_q_each = int(len(waypoints[0]) / len(sub_extend_fns))
    sub_waypoints, sub_paths = [], []
    for r in range(len(sub_extend_fns)):
        sub_waypoints.append([p[len_q_each * r: len_q_each * (r + 1)] for p in waypoints])
        sub_paths.append(list(flatten(sub_extend_fns[r](q1, q2) for q1, q2 in get_pairs(sub_waypoints[r]))))

    merged_sub_paths = []
    for j in range(get_max_length_list(sub_paths)):
        q = ()
        for r in range(len(sub_extend_fns)):
            if len(sub_paths[r]) <= j:
                q += sub_paths[r][len(sub_paths[r]) - 1]
            else:
                q += sub_paths[r][j]
        merged_sub_paths.append(q)
    return merged_sub_paths


def smooth_collision_check(q, robots, sub_collision_fns):
    len_q_each = int(len(q) / len(sub_collision_fns))
    for r in range(len(sub_collision_fns)):
        if sub_collision_fns[r](q[len_q_each * r : len_q_each * (r + 1)]):
            return True
    for r in range(len(sub_collision_fns) - 1):
        for r_prime in range(r + 1, len(sub_collision_fns)):
            if pairwise_collision(robots[r], robots[r_prime]):
                return True
    return False


##### dRRT*

class OptimalNode(object):

    def __init__(self, config, parent=None, d=0, num_robots=1, subprob_id=[], path=[], attachments=[]):
        self.config = config
        self.parent = parent
        self.children = set()
        self.d = d
        self.num_robots = num_robots
        self.subprob_id = copy.deepcopy(subprob_id)
        if parent is not None:
            self.cost = parent.cost + d
            self.parent.children.add(self)
        else:
            self.cost = d
        self.sub_config = []
        for r in range(self.num_robots):
            self.sub_config.append(self.config[r*int(len(config)/num_robots):
                                               (r+1)*int(len(config)/num_robots)])
        self.sub_local_paths = path
        self.attachments = attachments

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def rewire(self, parent, d, path):
        self.parent.children.remove(self)
        self.parent = parent
        self.parent.children.add(self)
        self.d = d
        self.sub_local_paths = path
        self.update()

    def update(self):
        self.cost = self.parent.cost + self.d
        for n in self.children:
            n.update()

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def draw(self, env):
        # https://github.mit.edu/caelan/lis-openrave
        from manipulation.primitives.display import draw_node, draw_edge
        color = apply_alpha(BLUE if self.solution else RED, alpha=0.5)
        self.node_handle = draw_node(env, self.config, color=color)
        if self.parent is not None:
            self.edge_handle = draw_edge(
                env, self.config, self.parent.config, color=color)

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.config) + ')'
    __repr__ = __str__


def compute_local_dist(roadmaps, q_old, q_new):
    q_old = get_sub_q_list(q_old, len(roadmaps))
    q_new = get_sub_q_list(q_new, len(roadmaps))
    dist = 0
    for r in range(len(roadmaps)):
        dist += roadmaps[r].sub_distance_fn(q_old[r], q_new[r])
    return dist


def get_sub_q_list(q, num_robots):
    sub_q = []
    num_elems = int(len(q)/num_robots)
    for i in range(num_robots):
        sub_q.append(q[i*num_elems:i*num_elems+num_elems])
    return sub_q


def get_min_heuristic_vertex(roadmap, q_near, heuristics):
    min_heuristic_val = heuristics[roadmap.vertices[q_near]]
    min_heuristic_vertex = q_near
    for v in roadmap.vertices[q_near].edges.keys():
        if heuristics[v] < min_heuristic_val:
            min_heuristic_val = heuristics[v]
            min_heuristic_vertex = v.q
    return min_heuristic_vertex


def get_random_neighbor_vertex(roadmap, q_near):
    candidate_vertices = [q_near]
    for v in roadmap.vertices[q_near].edges.keys():
        candidate_vertices.append(v.q)
    return candidate_vertices[random.randrange(len(candidate_vertices))]


def apply_order_constraints(q_new, q_near, r, order_constraint, sub_goals, subprob_id):
    if order_constraint[int(subprob_id[r] / 3)]:    # TODO: 3 is hardcoded for three subactions: base motion, grasp motion, post-grasp motion
        sub_goal = sub_goals[r]
        if q_new != sub_goal: return q_new
        constraints = order_constraint[int(subprob_id[r] / 3)]
        for constraint in constraints:
            for constraint_r in constraint.keys():
                if subprob_id[constraint_r] <= constraint[constraint_r]:
                    # print('order_constraint applied for robot ', r)
                    q_new = q_near
                    return q_new
    return q_new


def is_violate_order_constraints(order_constraints, subprob_id):
    for r in range(len(order_constraints)):
        if order_constraints[r][int(subprob_id[r] / 3)]:
            constraints = order_constraints[r][int(subprob_id[r] / 3)]
            for constraint in constraints:
                for constraint_r in constraint.keys():
                    if subprob_id[constraint_r] <= constraint[constraint_r]:
                        return True
    return False


def get_neighbor_vertices(nodes, roadmaps, q_new, subprob_id):
    neighbor_vertices = []
    q_new = get_sub_q_list(q_new, len(roadmaps))
    for n in nodes:
        if n.subprob_id != subprob_id: continue
        is_found_in_node = True
        is_remain = [False for _ in range(len(roadmaps))]
        for r in range(len(roadmaps)):
            is_found_in_roadmap = False
            if n.sub_config[r] == roadmaps[r].vertices[q_new[r]].q:
                is_remain[r] = True
                continue
            for v in roadmaps[r].vertices[q_new[r]].edges.keys():
                if n.sub_config[r] == v.q:
                    is_found_in_roadmap = True
                    break
            if not is_found_in_roadmap:
                is_found_in_node = False
                break
        if all(is_remain) == True: continue
        if not is_found_in_node:
            continue
        else:
            neighbor_vertices.append(n.config)
    return neighbor_vertices


def get_q_best(nodes, roadmaps, num_robots, drrt_collision_fn, neighbor_nodes, q_new, subprob_id):
    min_dist = float('inf')
    q_best, local_dist, local_paths = None, None, None
    for n in neighbor_nodes:
        paths = get_local_paths(roadmap=roadmaps, num_robots=num_robots,
                                start_configs=get_sub_q_list(n, num_robots),
                                target_configs=get_sub_q_list(q_new, num_robots))
        if is_local_collision(paths, drrt_collision_fn): continue
        node = get_node_from_tree(nodes, n, subprob_id)
        dist = compute_local_dist(roadmaps, n, q_new)
        if dist + node.cost < min_dist:
            min_dist = dist + node.cost
            q_best = node.config
            local_dist = dist
            local_paths = paths
    return q_best, local_dist, local_paths


def is_goal_in_tree(nodes, goal, roadmaps, subprob_id):
    for r in range(len(roadmaps)):
        if len(roadmaps[r]) > subprob_id[r] + 1:
            return False
    for n in nodes:
        if n.subprob_id == subprob_id:
            if n.config == goal:
                return True
    return False


def is_q_new_in_subgoal(roadmaps, q_new, goal):
    sub_q_new = get_sub_q_list(q_new, len(roadmaps))
    sub_goal = get_sub_q_list(goal, len(roadmaps))
    for i in range(len(sub_q_new)):
        if sub_q_new[i] == sub_goal[i]:
            return True
    return False


def get_node_from_tree(nodes, q, subprob_id):
    for n in nodes:
        if n.subprob_id == subprob_id:
            if n.config == q:
                return n
    raise SystemExit('ERROR: Node does not exist in the tree.')


def rewire(nodes, roadmaps, num_robots, q_parent, q_child):
    if get_node_from_tree(nodes, q_child).subprob_id == get_node_from_tree(nodes, q_parent).subprob_id:
        if get_node_from_tree(nodes, q_child).parent.config != q_parent:
            d = compute_local_dist(roadmaps, q_parent, q_child)
            local_paths = get_local_paths(roadmap=roadmaps, num_robots=num_robots,
                                          start_configs=get_sub_q_list(q_parent, num_robots),
                                          target_configs=get_sub_q_list(q_child, num_robots))
            get_node_from_tree(nodes, q_child).rewire(get_node_from_tree(nodes, q_parent), d, local_paths)


def get_heuristic_val(roadmaps, q, heuristic_vals):
    heuristic_val = 0
    q = get_sub_q_list(q, len(roadmaps))
    for r in range(len(roadmaps)):
        heuristic_val += heuristic_vals[r][roadmaps[r].vertices[q[r]]]
    return heuristic_val


def get_goals(array, dim):
    goals = ()
    for i in range(len(array)):
        goals += array[i][-dim:]
    return goals


def get_substarts_subgoals(array, subprob_id, dim):
    sub_array = ()
    for i in range(len(array)):
        sub_array += array[i][dim * subprob_id[i]:dim * (subprob_id[i] + 1)]
    return sub_array


def get_subprob(array, subprob_id):
    sub_array = []
    for i in range(len(array)):
        sub_array.append(array[i][subprob_id[i]])
    return sub_array


def get_subattachments(array, subprob_id, nodes):
    # remember that the local path is the path reaching to the current node, so attachment must use the previous node information
    sub_array = [None for _ in range(len(array))]
    for r in range(len(array)):
        if nodes[-1].subprob_id[r] == subprob_id[r]:
            if array[r][subprob_id[r]].attachments:
                sub_array[r] = array[r][subprob_id[r]].attachments[0]
        else:
            if array[r][subprob_id[r] - 1].attachments:
                sub_array[r] = array[r][subprob_id[r] - 1].attachments[0]
    return sub_array


##### Debugging tools

def debug_2d_sub_roadmap(roadmap, start, goal):
    _roadmap = []
    for v in roadmap.vertices.keys():
        if (v[0] == start[0] and v[1] == start[1]) or (v[0] == goal[0] and v[1] == goal[1]):
            plt.plot(v[0], v[1], 'ro', markersize=10)
        else: plt.plot(v[0], v[1], 'ko', markersize=10)
        _roadmap.append((round(v[0], 2), round(v[1], 2)))
        for e in roadmap[v].edges.keys():
            plt.plot([v[0], e.q[0]], [v[1], e.q[1]], 'k')
    # plt.savefig('save/roadmap.png')
    # plt.show()
    plt.clf()


def debug_2d_sub_sampling(index, robot_id, roadmap, tree, sample, q_near, q_new, q_new_candidates, angles, q_new_id):
    plt.plot(sample[0], sample[1], 'bD')
    _roadmap = []
    for v in roadmap.vertices.keys():
        plt.plot(v[0], v[1], 'ko')
        _roadmap.append((round(v[0], 2), round(v[1], 2)))
        for e in roadmap[v].edges.keys():
            plt.plot([v[0], e.q[0]], [v[1], e.q[1]], 'k')
    _tree = []
    for v in get_sub_nodes(tree, robot_id):
        plt.plot(v[0], v[1], 'gs', markersize=8)
        _tree.append((round(v[0], 2), round(v[1], 2)))
    plt.plot(q_near[0], q_near[1], 'ro')
    _q_new_candidates = []
    for i in range(len(q_new_candidates)):
        plt.plot(q_new_candidates[i].q[0], q_new_candidates[i].q[1], 'cs')
        _q_new_candidates.append((round(q_new_candidates[i].q[0], 2), round(q_new_candidates[i].q[1], 2)))
    plt.plot(q_new[0], q_new[1], 'ms')
    sample = ['%.2f' % elem for elem in sample]
    q_near = ['%.2f' % elem for elem in q_near]
    q_new = ['%.2f' % elem for elem in q_new]
    angles = ['%.2f' % elem for elem in angles]
    print('index: {}\nsample: {}, q_near: {}, q_new: {},\nroadmap vertices: {},\n'
          'tree vertices: {},\nq_new_id: {},\nq_new_candidates: {}, angles: {},\n'.
          format(index, sample, q_near, q_new, _roadmap, _tree, q_new_id, _q_new_candidates, angles))
    # plt.savefig('save/debug_{}_{}.png'.format(robot_id, index))
    # plt.show()
    plt.clf()