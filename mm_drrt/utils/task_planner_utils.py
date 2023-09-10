import numpy as np
import networkx as nx
import time
from mm_drrt.planner.drrt_star import dRRTStar


class Action:
    def __init__(self, type, robot, m_obj, from_f_obj, to_f_obj, pre_constraints, post_constraints, action_name=None):
        self.type = type    # transit or transfer
        self.robot = robot
        self.m_obj = m_obj
        self.from_f_obj = from_f_obj
        self.to_f_obj = to_f_obj
        self.name = action_name    # e.g. 'a0' given as input

        self.start = {'place': None, 'grasp': None, 'base': None, 'approach_conf': None, 'grasp_conf': None}
        self.goal = {'place': None, 'grasp': None, 'base': None, 'approach_conf': None, 'grasp_conf': None}

        self.placement_samples = []
        self.order_constraints = {'pre': pre_constraints, 'post': post_constraints}
        self.obstacles = {'start': {'objs': [], 'poses': []}, 'goal': {'objs': [], 'poses': []}}

    def __repr__(self):
        return self.name + '_' + self.type


class Fixedobj:
    '''
    grasp_objs: transit action to grasp an obj (step 2)
    add_objs: transfer action to place an obj (step 1)
    remove_objs: transfer action to clear an obj (step 1)
    '''
    def __init__(self, env, f_obj, init_m_objs, num_samples):
        self.env = env
        self.f_obj = f_obj
        self.grasp_actions = {}
        self.add_actions = {}
        self.remove_actions = {}
        self.m_objs = set()
        self.obj_orders = {}
        self.init_m_objs = init_m_objs
        self.num_samples = num_samples
        self.collisions = []

    '''
    Adding actions related to this fixed obj
    '''
    def add_action(self, action, type):
        if type == 'grasp': # when action is 'transit' and to_f_obj
            self.grasp_actions[action.name] = action
        elif type == 'add': # when action is 'transfer' and to_f_obj
            self.add_actions[action.name] = action
            self.m_objs.add(action.m_obj)
        else:   # 'remove' # when action is 'transfer' and from_f_obj
            self.remove_actions[action.name] = action
            self.m_objs.add(action.m_obj)

    '''
    Setting obj orders
    '''
    def set_obj_orders(self, obj_orders):
        for m_obj in self.m_objs:
            action_sequence = []
            for a in obj_orders[m_obj]:
                if a in self.add_actions:
                    action_sequence.append((a, 'add'))
                elif a in self.remove_actions:
                    action_sequence.append((a, 'remove'))
            if action_sequence:
                self.obj_orders[m_obj] = action_sequence
        self._initialize_placement_sampling()

    '''
    Initialize placement sampling
    '''
    def _initialize_placement_sampling(self):
        self.stationary_m_objs, self.remove_m_objs, self.add_m_objs, \
        self.remove_then_add_m_objs, self.add_then_remove_m_objs = classify_obj_types(self.obj_orders, self.init_m_objs)

        # placement sampling
        placement_sampling_for_add_actions(self.env,  self.f_obj, self.num_samples, self.obj_orders, self.add_actions,
                                           self.add_m_objs, self.remove_then_add_m_objs, self.add_then_remove_m_objs)
        # TODO: remove_m_objs may have add actions, e.g. remove->add->remove. for now, we do not apply sampling but use
        #  the initial pose since it does not hurt feasibility by removing this object entirely first before adding its collision objects
        self.remove_objs_poses = [self.env.m_objs_init_placements[remove_m_obj] for remove_m_obj in self.remove_m_objs]

        # collision checking
        self.check_indices = [0] * len(self.remove_then_add_m_objs + self.add_then_remove_m_objs + self.add_m_objs)

    '''
    Step 1: finding placements of objs that will be added to this fixed obj
    '''
    def is_refine_placements(self):
        is_feasible, collisions, init_collisions = is_placement_collision(self.env, self.check_indices,
                                                                          self.obj_orders, self.add_actions, self.remove_objs_poses,
                                                                          self.stationary_m_objs, self.remove_m_objs, self.add_m_objs,
                                                                          self.remove_then_add_m_objs, self.add_then_remove_m_objs)

        if is_feasible:    # collision free placements are found. move onto step 2
            if collisions:  # collisions with objs that will be removed
                for collision in collisions:
                    add_order_constraints(collision, 'collision', self.obj_orders, self.add_actions, self.remove_actions)
            if init_collisions: # initial pose of remove_then_add_m_objs collides with objs that will be added
                for collision in init_collisions:
                    add_order_constraints(collision, 'init_collision', self.obj_orders, self.add_actions, self.remove_actions)

            # set obstacles for add_actions and remove_actions
            set_obstacles(self.add_actions, self.remove_actions, self.check_indices, self.env, self.obj_orders, collisions, init_collisions,
                          self.stationary_m_objs, self.remove_m_objs, self.add_m_objs, self.remove_then_add_m_objs, self.add_then_remove_m_objs)
            # set placmenets to add actions
            set_placements_to_add_actions(self.check_indices, self.add_actions, self.obj_orders,
                                          self.add_m_objs, self.remove_then_add_m_objs, self.add_then_remove_m_objs)
            self._check_indices_update()
            return True
        else:
            self._check_indices_update()
            return False

    def _check_indices_update(self):
        self.check_indices = update_check_indices(self.check_indices, self.num_samples)
        if is_done_placement_collision_checking(self.check_indices):  # termination
            raise Exception('All samples tried but no path is found...')

    def __repr__(self):
        return 'fixed_obj_' + str(self.f_obj)

##################################################################
##################################################################

'''
utils used in the PlanSkeleton class
'''

def initialize_actions(plan, env, actions, init_order_constraints, f_objs, num_samples):
    found_f_objs = set()
    for a in plan.items():
        type = a[1][0]
        robot = a[1][1]
        m_obj = a[1][2]
        from_f_obj = a[1][3]
        to_f_obj = a[1][4]
        pre, post = [], []
        for constraint in init_order_constraints:
            for order, action in constraint.items():
                if action == a[0]:
                    if order == 'pre':
                        post.append(constraint['post'])
                    elif order == 'post':
                        pre.append(constraint['pre'])
        actions[a[0]] = Action(type, robot, m_obj, from_f_obj, to_f_obj, pre, post, action_name=a[0])

        if from_f_obj:
            if from_f_obj not in found_f_objs:
                found_f_objs.add(from_f_obj)
                if from_f_obj in env.m_obj_in_f_obj:
                    f_objs[from_f_obj] = Fixedobj(env, from_f_obj, env.m_obj_in_f_obj[from_f_obj], num_samples)
                else:
                    f_objs[from_f_obj] = Fixedobj(env, from_f_obj, set(), num_samples)
            f_objs[from_f_obj].add_action(actions[a[0]], 'remove')

        if to_f_obj:
            if to_f_obj not in found_f_objs:
                found_f_objs.add(to_f_obj)
                if to_f_obj in env.m_obj_in_f_obj:
                    f_objs[to_f_obj] = Fixedobj(env, to_f_obj, env.m_obj_in_f_obj[to_f_obj], num_samples)
                else:
                    f_objs[to_f_obj] = Fixedobj(env, to_f_obj, set(), num_samples)
            if type == 'transit':
                f_objs[to_f_obj].add_action(actions[a[0]], 'grasp')
            else:  # type == 'transfer'
                f_objs[to_f_obj].add_action(actions[a[0]], 'add')


def initialize_robot_plans(plan, env, actions, robot_plans):
    for r in env.robots.values():
        robot_plan = []
        for p in plan:
            if plan[p][1] == r:
                robot_plan.append(actions[p])
        robot_plans[r] = robot_plan


def subgoal_refinement(robot_plans, env, obj_orders, actions, use_debug=False):
    init_world = env.save_world()
    for robot in robot_plans:
        for id, action in enumerate(robot_plans[robot]):
            if id % 2 == 1: # place actions
                if not env.subgoal_sampling(robot, obj_orders[robot_plans[robot][id].m_obj], actions,
                                            robot_plans[robot][id], robot_plans[robot][id].m_obj,
                                            robot_plans[robot][id].obstacles['start'], robot_plans[robot][id].obstacles['goal'],
                                            custom_limits=env.custom_limits, use_debug=use_debug):
                    env.restore_world(init_world)
                    return False
    env.restore_world(init_world)
    return True


def individual_path_computation(robot_plans, env, num_base_samples, num_arm_samples, roadmaps, heuristic_vals, use_debug=False):
    init_world = env.save_world()
    for robot in robot_plans:
        each_roadmaps, each_heuristic_vals = [], []
        for id, action in enumerate(robot_plans[robot]):
            if action.type != 'return':
                if id % 2 == 0: # pick actions
                    if id == 0:
                        action.start['base'] = env.get_base_conf(robot)
                    else:
                        action.start['base'] = robot_plans[robot][id - 1].goal['base']
                    action.start['place'] = robot_plans[robot][id + 1].start['place']
                    action.goal['base'] = robot_plans[robot][id + 1].start['base']
                    action.goal['grasp'] = robot_plans[robot][id + 1].start['grasp']
                    action.goal['place'] = robot_plans[robot][id + 1].start['place']
                    action.goal['approach_conf'] = robot_plans[robot][id + 1].start['approach_conf']
                    action.goal['grasp_conf'] = robot_plans[robot][id + 1].start['grasp_conf']

                    action.obstacles['start'] = robot_plans[robot][id + 1].obstacles['start']
                    action.obstacles['goal'] = robot_plans[robot][id + 1].obstacles['start']
                    result, r_each_roadmaps, r_each_heuristic_vals = env.compute_path(robot, action, action.m_obj,
                                                                                      num_base_samples, num_arm_samples,
                                                                                      type='transit', use_debug=use_debug)
                else:
                    result, r_each_roadmaps, r_each_heuristic_vals = env.compute_path(robot, action, action.m_obj,
                                                                                      num_base_samples, num_arm_samples,
                                                                                      type='transfer', use_debug=use_debug)
            else:   # action.type == 'return'
                action.start['base'] = env.get_base_conf(robot)
                action.goal['base'] = env.get_init_base_conf(robot)
                result, r_each_roadmaps, r_each_heuristic_vals = env.compute_path(robot, action, action.m_obj,
                                                                                  num_base_samples, num_arm_samples,
                                                                                  type='return', use_debug=use_debug)
            if result:
                each_roadmaps += r_each_roadmaps
                each_heuristic_vals += r_each_heuristic_vals
                if use_debug:
                    print('In Step 3: '+ str(action) + ' is refined')
            else:
                return False, [], []
        roadmaps.append(each_roadmaps)
        heuristic_vals.append(each_heuristic_vals)
    env.restore_world(init_world)
    return True, roadmaps, heuristic_vals


def composite_path_computation(env, robots, roadmaps, heuristic_vals, order_constraints, drrt_num_iters, drrt_time_limit):
    init_world = env.save_world()
    joints = env.get_joints(robots)
    tensor_roadmap = dRRTStar(robots, joints, roadmaps, order_constraints, num_robots=len(robots), heuristic_vals=heuristic_vals)
    start_time = time.time()
    composite_path = tensor_roadmap.grow(num_iters=drrt_num_iters, start_time=start_time, time_limit=drrt_time_limit)
    env.restore_world(init_world)
    return composite_path


def assign_order_constraints(robot_plans):
    constraints = []
    for r in robot_plans.keys():
        constraint = ()
        for a in robot_plans[r]:
            pre_set = a.order_constraints['pre']
            pre_index = []
            if pre_set:
                for i, nr in enumerate(robot_plans.keys()):
                    if r == nr: continue
                    for j in range(len(robot_plans[nr])):
                        if robot_plans[nr][j].name in a.order_constraints['pre']:
                            pre_index.append({i: 3 * (j + 1) - 1})    # each action consists of three subactions: base motion, grasp motion, post-grasp motion
                            break
            constraint += (pre_index,)
        constraints.append(constraint)
    return constraints


##################################################################

'''
utils used in the Fixedobj class
'''

def classify_obj_types(obj_orders, init_m_objs):
    remove_m_objs = []
    add_m_objs = []
    remove_then_add_m_objs = []
    add_then_remove_m_objs = []
    for m_obj in obj_orders:
        if obj_orders[m_obj][0][1] == 'remove':
            if obj_orders[m_obj][-1][1] == 'remove':
                remove_m_objs.append(m_obj)
            else:   # obj_orders[m_obj][-1][1] == 'add'
                remove_then_add_m_objs.append(m_obj)
        else:   # obj_orders[m_obj][0][1] == 'add'
            if obj_orders[m_obj][-1][1] == 'remove':
                add_then_remove_m_objs.append(m_obj)
            else:   # obj_orders[m_obj][-1][1] == 'add'
                add_m_objs.append(m_obj)
    stationary_m_objs = list(init_m_objs - (set(remove_m_objs) | set(remove_then_add_m_objs)))

    return stationary_m_objs, remove_m_objs, add_m_objs, remove_then_add_m_objs, add_then_remove_m_objs


def placement_sampling(env, m_obj, f_obj, num_samples, init_pose, action):
    if init_pose:
        placements = [init_pose]
        placements = placements + env.placement_sample(m_obj, f_obj, num_samples - 1)
    else:
        placements = env.placement_sample(m_obj, f_obj, num_samples)
    action.placement_samples = placements


def placement_sampling_for_add_actions(env, f_obj, num_samples, obj_orders, add_actions,
                                       add_m_objs, remove_then_add_m_objs, add_then_remove_m_objs):
    for remove_then_add_m_obj in remove_then_add_m_objs:
        add_action = None
        for action in obj_orders[remove_then_add_m_obj]:
            if action[1] == 'add':
                add_action = action[0]
                break
        placement_sampling(env, remove_then_add_m_obj, f_obj, num_samples,
                           env.m_objs_init_placements[remove_then_add_m_obj], add_actions[add_action])

    for add_m_obj in add_m_objs + add_then_remove_m_objs:
        add_action = None
        for action in obj_orders[add_m_obj]:
            if action[1] == 'add':
                add_action = action[0]
                break
        placement_sampling(env, add_m_obj, f_obj, num_samples, None, add_actions[add_action])


def is_placement_collision(env, check_indices, obj_orders, add_actions, remove_objs_poses,
                           stationary_m_objs, remove_m_objs, add_m_objs, remove_then_add_m_objs, add_then_remove_m_objs):
    saved_world = env.save_world()
    objs_poses = []
    for id, add_m_obj in enumerate(remove_then_add_m_objs + add_then_remove_m_objs + add_m_objs):
        add_action = None
        for action in obj_orders[add_m_obj]:
            if action[1] == 'add':
                add_action = action[0]
                break
        objs_poses.append(add_actions[add_action].placement_samples[check_indices[id]])

    is_feasible, collisions = env.is_placement_collision(objs_poses + remove_objs_poses, stationary_m_objs,
                                                         remove_m_objs, add_m_objs, remove_then_add_m_objs, add_then_remove_m_objs)

    if is_feasible:    # may or may not collide with objs that will be removed
        init_collisions = []    # check if initial pose of remove_then_add_m_objs is in collision with objs that will be added
        for id, m_obj in enumerate(remove_then_add_m_objs):
            if check_indices[id] > 0:
                add_action = None
                for action in obj_orders[m_obj]:
                    if action[1] == 'add':
                        add_action = action[0]
                        break
                init_obj_pose = add_actions[add_action].placement_samples[0]  # placement_samples[0] outputs the init pose of remove_then_add_m_objs
                cur_obj_pose = add_actions[add_action].placement_samples[check_indices[id]]
                init_collisions = env.placement_collision_with_remove_then_add_m_objs(init_obj_pose, cur_obj_pose,
                                                                                      id, m_obj,
                                                                                      init_collisions, add_m_objs,
                                                                                      add_then_remove_m_objs,
                                                                                      remove_then_add_m_objs)
        env.restore_world(saved_world)
        return True, collisions, init_collisions
    else:   # infeasible
        env.restore_world(saved_world)
        return False, [], []    # is_feasible, collisions


def update_check_indices(check_indices, num_samples):
    for i in reversed(range(len(check_indices))):
        new_index = check_indices[i] + 1
        if new_index == num_samples:
            check_indices[i] = 0
        else:
            check_indices[i] = new_index
            break
    return check_indices


def is_done_placement_collision_checking(check_indices):
    if check_indices:
        for index in check_indices:
            if index != 0:
                return False
        return True
    else: return False


def add_order_constraints(collision, collision_type, obj_orders, add_actions, remove_actions):
    # TODO: for simplicity, for now if add_obj includes multiple add-remove pairs of actions and remove_obj also includes
    #  as such, we add add_obj after remove_obj is entirely removed. fix this if we want more optimized task plans
    add_obj, remove_obj = collision
    if collision_type == 'collision':   # collisions with remove_m_objs, add_then_remove_m_objs
        for action in reversed(obj_orders[remove_obj]):
            if action[1] == 'remove':
                remove_action = action[0]
                break
    elif collision_type == 'init_collision':    # collisions with remove_then_add_m_obj
        remove_action = obj_orders[remove_obj][0][0]

    for action in obj_orders[add_obj]:
        if action[1] == 'add':
            add_action = action[0]
            break
    add_actions[add_action].order_constraints['pre'].append(remove_actions[remove_action])
    remove_actions[remove_action].order_constraints['next'].append(add_actions[add_action])


def set_obstacles(add_actions, remove_actions, check_indices, env, obj_orders, collisions, init_collisions,
                  stationary_m_objs, remove_m_objs, add_m_objs, remove_then_add_m_objs, add_then_remove_m_objs):
    # find all obstacles' ids and poses
    all_obst, all_obst_poses = [], []
    all_obst, all_obst_poses = find_obj_pose_from_env(env, stationary_m_objs, all_obst, all_obst_poses)

    # remove_then_add_m_objs + add_m_objs
    all_obst, all_obst_poses = find_obj_pose_from_set(obj_orders, add_actions, check_indices, remove_then_add_m_objs,
                                                      all_obst, all_obst_poses)
    all_obst, all_obst_poses = find_obj_pose_from_set(obj_orders, add_actions, check_indices, add_m_objs,
                                                      all_obst, all_obst_poses,
                                                      pre_id_len=len(remove_then_add_m_objs) + len(add_then_remove_m_objs))

    # remove_m_objs + add_then_remove_m_objs
    all_obst, all_obst_poses = find_obj_pose_from_env(env, remove_m_objs, all_obst, all_obst_poses)
    all_obst, all_obst_poses = find_obj_pose_from_set(obj_orders, add_actions, check_indices, add_then_remove_m_objs,
                                                      all_obst, all_obst_poses, pre_id_len=len(remove_then_add_m_objs))

    # find if any of remove_then_add_m_objs were moved i.e. check_indices > 0
    remove_then_add_obst, remove_then_add_obst_poses = [], []
    for id, remove_then_add_m_obj in enumerate(remove_then_add_m_objs):
        if check_indices[id] > 0:
            for m_obj in env.m_objs_init_placements:
                if remove_then_add_m_obj == m_obj:
                    remove_then_add_obst.append(m_obj)
                    remove_then_add_obst_poses.append(env.m_objs_init_placements[m_obj])
                    break

    # set the relevant obstacles to add_actions and remove_actions
    for m_obj in obj_orders:
        # add actions
        ignore_from_remove_m_objs = []
        ignore_from_remove_then_add_obst = []
        for action in obj_orders[m_obj]:
            if action[1] == 'add':
                ignore_from_remove_m_objs, ignore_from_remove_then_add_obst = \
                    find_ignore_objs(add_actions, action, collisions, init_collisions, ignore_from_remove_m_objs,
                                     ignore_from_remove_then_add_obst, type='add')

                obst, obst_poses = [], []
                obst, obst_poses = find_obst_poses(all_obst, all_obst_poses, obst, obst_poses, m_obj, ignore_from_remove_m_objs)
                obst, obst_poses = find_obst_poses(remove_then_add_obst, remove_then_add_obst_poses, obst,
                                                   obst_poses, m_obj, ignore_from_remove_then_add_obst)
                add_actions[action[0]].obstacles['goal']['objs'] = obst
                add_actions[action[0]].obstacles['goal']['poses'] = obst_poses

        # remove actions
        ignore_from_add_m_objs = []
        ignore_from_remove_then_add_obst = []
        for action in reversed(obj_orders[m_obj]):
            if action[1] == 'remove':
                ignore_from_add_m_objs, ignore_from_remove_then_add_obst = \
                    find_ignore_objs(remove_actions, action, collisions, init_collisions, ignore_from_add_m_objs,
                                     ignore_from_remove_then_add_obst, type='remove')

                obst, obst_poses = [], []
                obst, obst_poses = find_obst_poses(all_obst, all_obst_poses, obst, obst_poses, m_obj,
                                                   ignore_from_add_m_objs)
                obst, obst_poses = find_obst_poses(remove_then_add_obst, remove_then_add_obst_poses, obst,
                                                   obst_poses, m_obj, ignore_from_remove_then_add_obst)
                remove_actions[action[0]].obstacles['start']['objs'] = obst
                remove_actions[action[0]].obstacles['start']['poses'] = obst_poses


def set_placements_to_add_actions(check_indices, add_actions, obj_orders,
                                  add_m_objs, remove_then_add_m_objs, add_then_remove_m_objs):
    for id, add_m_obj in enumerate(remove_then_add_m_objs + add_then_remove_m_objs + add_m_objs):
        for action in obj_orders[add_m_obj]:
            if action[1] == 'add':
                add_actions[action[0]].goal['place'] = add_actions[action[0]].placement_samples[check_indices[id]]


def find_obj_pose_from_env(env, m_objs, obst, obst_poses):
    for m_o in m_objs:
        for m_obj in env.m_objs_init_placements:
            if m_o == m_obj:
                obst.append(m_obj)
                obst_poses.append(env.m_objs_init_placements[m_obj])
                break
    return obst, obst_poses


def find_obj_pose_from_set(obj_orders, add_actions, check_indices, m_objs, obst, obst_poses, pre_id_len=0):
    for id, m_obj in enumerate(m_objs):
        for action in obj_orders[m_obj]:
            if action[1] == 'add':
                obst.append(m_obj)
                obst_poses.append(add_actions[action[0]].placement_samples[check_indices[pre_id_len + id]])
                break
    return obst, obst_poses


def find_obst_poses(input_obst, input_obst_poses, obst, obst_poses, m_obj, ignore_from_m_objs):
    for id in range(len(input_obst)):
        if input_obst[id] == m_obj: continue
        is_in_ignore_from_m_objs = False
        for ignore_from_m_obj in ignore_from_m_objs:
            if ignore_from_m_obj == input_obst[id]:
                is_in_ignore_from_m_objs = True
                break
        if not is_in_ignore_from_m_objs:
            obst.append(input_obst[id])
            obst_poses.append(input_obst_poses[id])
    return obst, obst_poses


def find_ignore_objs(actions, action, collisions, init_collisions, ignore_from_m_objs, ignore_from_remove_then_add_obst, type=None):
    if type == 'add':
        const_actions = actions[action[0]].order_constraints['pre']
    elif type == 'remove':
        const_actions = actions[action[0]].order_constraints['post']
    for const_action in const_actions:
        for collision in collisions:  # collisions with remove_m_objs, add_then_remove_m_objs
            if type == 'add':
                obj = collision[1]
            elif type == 'remove':
                obj = collision[0]
            if const_action.m_obj == obj:
                ignore_from_m_objs.append(const_action.m_obj)
                break
        for init_collision in init_collisions:  # collisions with remove_then_add_m_obj
            if type == 'add':
                obj = init_collision[1]
            elif type == 'remove':
                obj = init_collision[0]
            if const_action.m_obj == obj:
                ignore_from_remove_then_add_obst.append(const_action.m_obj)
                break
    return ignore_from_m_objs, ignore_from_remove_then_add_obst