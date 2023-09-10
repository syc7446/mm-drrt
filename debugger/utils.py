from __future__ import print_function

import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance

from external.pybullet_planning.pybullet_tools.utils import get_sample_fn, get_distance_fn, \
    get_extend_fn, get_collision_fn, get_joint_positions, check_initial_end, pairwise_collision, \
    get_moving_links, can_collide, CollisionPair, parse_body, cached_fn, get_buffered_aabb, \
    set_joint_positions, aabb_overlap, product, MAX_DISTANCE


def sub_plan_joint_motion(roadmap, robot, joints, start_conf, end_conf, obstacles=[], attachments=[],
                          self_collisions=True, disabled_collisions=set(),
                          weights=None, resolutions=None, max_distance=MAX_DISTANCE,
                          use_aabb=False, cache=True, custom_limits={}, use_drrt_star=False,
                          use_debug_plot=False, **kwargs):

    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sub_sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    sub_distance_fn = get_distance_fn(robot, joints, weights=weights)
    sub_extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    sub_collision_fn = get_collision_fn(robot, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                        custom_limits=custom_limits, max_distance=max_distance,
                                        use_aabb=use_aabb, cache=cache)
    roadmap.sub_sample_fn = sub_sample_fn
    roadmap.sub_distance_fn = sub_distance_fn
    roadmap.sub_extend_fn = sub_extend_fn
    roadmap.sub_collision_fn = sub_collision_fn
    roadmap.distance_fn = sub_distance_fn
    roadmap.initial_conf = start_conf
    roadmap.final_conf = end_conf

    return roadmap


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


def get_sub_q_near(nodes, roadmaps, sub_samples):
    sub_q_near = []
    for r in range(len(roadmaps)):
        sub_q_near.append(nodes[get_max_dist_node_index(nodes, roadmaps, sub_samples)].sub_config[r])
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
    # for i in range(max([len(paths[r]) for r in range(len(paths))])):
    #     configs = []
    #     for r in range(len(paths)):
    #         if i >= len(paths[r]): configs.append(paths[r][-1])
    #         else: configs.append(paths[r][i])
    #     for r in range(len(paths)):
    #         if drrt_collision_fn[r](configs): return True
    init_configs, mid_configs, final_configs = [], [], []
    for r in range(len(paths)):
        init_configs.append(paths[r][0])
        mid_configs.append(paths[r][int(len(paths[r]) / 2)])
        final_configs.append(paths[r][-1])
    for r in range(len(paths)):
        if drrt_collision_fn[r](init_configs): return True
        if drrt_collision_fn[r](mid_configs): return True
        if drrt_collision_fn[r](final_configs): return True
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


def get_max_dist_node_index(nodes, roadmaps, goal):
    dist = []
    for n in nodes:
        sub_dist = 0
        for r in range(len(roadmaps)):
            sub_dist += roadmaps[r].sub_distance_fn(n.sub_config[r], goal[r])
        dist.append(sub_dist)
    return np.argmin(dist)


def connect_to_target(roadmap=[], num_robots=1, start_configs=[], target_configs=[]):
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


def is_duplicate(list_elems, new_elem):
    for elem in list_elems:
        if elem.config == new_elem:
            return True
    return False


def get_parent_node_index(nodes, q_near):
    near_node = ()
    for q in q_near:
        near_node += q
    for i in range(len(nodes)):
        if nodes[i].config == near_node: return i


##### dRRT*

class OptimalNode(object):

    def __init__(self, config, parent=None, d=0, num_robots=1, path=[]):
        self.config = config
        self.parent = parent
        self.children = set()
        self.d = d
        self.num_robots = num_robots
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


def get_neighbor_vertices(nodes, roadmaps, q_new):
    neighbor_vertices = []
    q_new = get_sub_q_list(q_new, len(roadmaps))
    for n in nodes:
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


def is_goal_in_tree(nodes, goal):
    for n in nodes:
        if n.config == goal:
            return True
    return False


def get_node_from_tree(nodes, q):
    for n in nodes:
        if n.config == q:
            return n
    raise SystemExit('ERROR: Node does not exist in the tree.')


def rewire(nodes, roadmaps, num_robots, q_parent, q_child):
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