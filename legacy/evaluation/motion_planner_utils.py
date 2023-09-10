import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import random
import math
import copy

from external.pybullet_planning.pybullet_tools.ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics
from external.pybullet_planning.pybullet_tools.pr2_utils import get_gripper_link, get_arm_joints, arm_conf, open_arm, \
    get_aabb, get_disabled_collisions, get_group_joints, learned_pose_generator, create_gripper, PR2_GROUPS, GET_GRASPS, \
    get_top_grasps, get_side_grasps, compute_grasp_width, TOP_HOLDING_LEFT_ARM, SIDE_HOLDING_LEFT_ARM
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
    get_moving_links, can_collide, CollisionPair, parse_body, cached_fn, get_buffered_aabb, set_joint_positions, \
    aabb_overlap, product

from legacy.evaluation.prm import prm


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


def base_motion(robot, base_start, base_goal, teleport=False, obstacles=[], attachments=[], custom_limits={}, num_samples=100):
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
            roadmap, heuristic_val = sub_plan_joint_motion(robot, base_joints, base_goal, obstacles=obstacles,
                                                           attachments=attachments, disabled_collisions=disabled_collisions,
                                                           resolutions=resolutions, custom_limits=custom_limits,
                                                           num_samples=num_samples)
        if not roadmap: set_joint_positions(robot, base_joints, base_start)
        return roadmap, heuristic_val


def get_ir_sampler(robot, gripper, custom_limits={}, max_attempts=25, collisions=True, collision_objs=[], num_samples=100, learned=True):
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


def get_ik_fn(robot, custom_limits={}, collisions=True, collision_objs=[], num_samples=100, teleport=False):
    robot = robot
    obstacles = collision_objs if collisions else []
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
        grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose,
                                            custom_limits=custom_limits)  # , upper_limits=USE_CURRENT)
        # nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):  # [obj]
            # print('Grasp IK failure', grasp_conf)
            # if grasp_conf is not None:
            #    print(grasp_conf)
            #    #wait_if_gui()
            return None
        # approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
        #                                       upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose,
                                               custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            # print('Approach IK failure', approach_conf)
            # wait_if_gui()
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(robot, arm)
        attachments = {attachment.child: attachment}
        roadmap = None
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.05 ** np.ones(len(arm_joints))
            set_joint_positions(robot, arm_joints, default_conf)
            roadmap, heuristic_val = sub_plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                                           obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                                           custom_limits=custom_limits, resolutions=resolutions,
                                                           num_samples=num_samples)
            approach_path = roadmap(default_conf, approach_conf)
            if approach_path is None:
                print('Approach path failure')
                return None
            set_joint_positions(robot, arm_joints, approach_conf)
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits=custom_limits, resolutions=resolutions / 2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd, attachments, gripper_pose[0], roadmap, )  # Only this line has been changed from the original code

    return fn


def get_ik_ir_gen(robot, gripper, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    ir_sampler = get_ir_sampler(robot, gripper, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_fn(robot, teleport=teleport, **kwargs)

    def gen(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
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


def expand_roadmap_dim(roadmap, heuristics, type, robot, arm):
    if type == 'base':
        base_joints = [joint_from_name(robot, name) for name in PR2_GROUPS['base']]
        base_positions = get_joint_positions(robot, base_joints)
        for v in roadmap.vertices:
            v.expand_dim(base_positions, type='base')
    elif type == 'arm':
        arm_joints = get_arm_joints(robot, arm)
        arm_positions = get_joint_positions(robot, arm_joints)
        for v in roadmap.vertices:
            v.expand_dim(arm_positions, type='arm')
    a = 1


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
        for r in range(len(paths)): # TODO: currently redundant collisions are checked. remove redundancy
            if drrt_collision_fn[r](configs): return True
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
            if search(collision_robot_index, r):
                configs.append(paths[r][0])
                continue
            if i >= len(paths[r]): configs.append(paths[r][-1])
            else: configs.append(paths[r][i])
        for r in range(len(paths)):
            if search(collision_robot_index, r): continue
            if drrt_collision_fn[r](configs): collision_robot_index.append(r)
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


def get_inter_robots_collision_fn(robots, joints, attachments=[], robot_id=0, num_robots=1,
                                  use_aabb=False, cache=False, max_distance=MAX_DISTANCE, **kwargs):
    moving_links = frozenset(link for link in get_moving_links(robots[robot_id], joints[robot_id])
                             if can_collide(robots[robot_id], link)) # TODO: propagate elsewhere
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [CollisionPair(robots[robot_id], moving_links)] + list(map(parse_body, attached_bodies))
    get_obstacle_aabb = cached_fn(get_buffered_aabb, cache=cache, max_distance=max_distance/2., **kwargs)

    def drrt_collision_fn(q, verbose=False):
        obstacles = []
        for r in range(num_robots):
            set_joint_positions(robots[r], joints[r], q[r])
            if r == robot_id: # TODO: implement for obstacle robots
                for attachment in attachments:
                    attachment.assign()
            else:
                obstacles.append(robots[r])
        #wait_for_duration(1e-2)
        get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance/2., **kwargs)

        for body1, body2 in product(moving_bodies, obstacles):
            if (not use_aabb or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))) \
                    and pairwise_collision(body1, body2, **kwargs):
                if verbose: print(body1, body2)
                return True
        return False
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
        return None
    roadmap, heuristic_val = prm(start_conf, end_conf, sub_distance_fn, sub_sample_fn, sub_extend_fn,
                                 sub_collision_fn, use_drrt_star=use_drrt_star, use_debug_plot=use_debug_plot,
                                 debug_roadmap_fn=debug_2d_sub_roadmap, **kwargs)
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
    for r in range(num_robots):
        if drrt_collision_fn[r](start_configs): return []
        if drrt_collision_fn[r](target_configs): return []
        configs = []
        for c in range(len(target_configs)):
            if c == r: configs.append(start_configs[c])
            configs.append(target_configs[c])
        if drrt_collision_fn[r](configs): return []
        configs = []
        for c in range(len(start_configs)):
            if c == r: configs.append(target_configs[c])
            configs.append(start_configs[c])
        if drrt_collision_fn[r](configs): return []

    local_paths = []
    for r in range(num_robots):
        local_path = list(roadmap[r].sub_extend_fn(start_configs[r], target_configs[r]))[:-1]
        if not local_path: local_path = [start_configs[r]]
        if any(roadmap[r].sub_collision_fn(q) for q in local_path):
            return []
        local_paths.append(local_path)
    return local_paths


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


##### dRRT*

class OptimalNode(object):

    def __init__(self, config, parent=None, d=0, num_robots=1, subprob_id=[], path=[]):
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


def get_neighbor_vertices(nodes, roadmaps, q_new, subprob_id):
    neighbor_vertices = []
    q_new = get_sub_q_list(q_new, len(roadmaps))
    for n in nodes:
        if n.subprob_id != subprob_id: continue
        is_found_in_node = True
        for r in range(len(roadmaps)):
            is_found_in_roadmap = False
            if n.sub_config[r] == roadmaps[r].vertices[q_new[r]].q:
                continue
            for v in roadmaps[r].vertices[q_new[r]].edges.keys():
                if n.sub_config[r] == v.q:
                    is_found_in_roadmap = True
                    break
            if not is_found_in_roadmap:
                is_found_in_node = False
                break
        if not is_found_in_node:
            continue
        else:
            neighbor_vertices.append(n.config)
    return neighbor_vertices


def get_q_best(nodes, roadmaps, num_robots, drrt_collision_fn, neighbor_nodes, q_new):
    min_dist = float('inf')
    q_best, local_dist, local_paths = None, None, None
    for n in neighbor_nodes:
        paths = get_local_paths(roadmap=roadmaps, num_robots=num_robots,
                                start_configs=get_sub_q_list(n, num_robots),
                                target_configs=get_sub_q_list(q_new, num_robots))
        if is_local_collision(paths, drrt_collision_fn): continue
        node = get_node_from_tree(nodes, n)
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


def get_node_from_tree(nodes, q):
    for n in nodes:
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