import numpy as np
import math
import time

from debugger.utils import OptimalNode, get_inter_robots_collision_fn, get_angle, \
    is_duplicate, get_sub_nodes, get_parent_node_index, get_local_paths, is_local_collision, \
    get_sub_q_near, connect_to_target, get_collision_free_paths_to_target, get_sub_q_list, \
    get_min_heuristic_vertex, get_random_neighbor_vertex, get_neighbor_vertices, compute_local_dist, \
    get_q_best, get_node_from_tree, rewire, get_heuristic_val, is_goal_in_tree, debug_2d_sub_sampling
from external.pybullet_planning.motion.motion_planners.utils import argmin
from external.pybullet_planning.pybullet_tools.utils import MAX_DISTANCE, join_paths, get_parent_dir, connect, disconnect, \
    disable_real_time, set_joint_positions, joint_from_name, wait_for_duration, wait_if_gui
from external.pybullet_planning.pybullet_tools.pr2_utils import PR2_GROUPS

from mm_drrt.utils.motion_planner_utils import get_max_length_list

class dRRTStar:

    def __init__(self, robots, joints, roadmaps, num_robots=1, heuristic_vals=[], radius=None,
                 attachments=[], max_distance=MAX_DISTANCE, use_aabb=False, cache=True):
        self.robots = robots
        self.joints = joints
        self.roadmaps = roadmaps
        self.num_robots = num_robots
        self.heuristic_vals = heuristic_vals
        self.radius = radius
        self.inter_robots_collision_fn = []
        self.sub_nodes = []

        self.start = ()
        self.goal = ()
        for r in range(self.num_robots):
            self.start += self.roadmaps[r].initial_conf
            self.goal += self.roadmaps[r].final_conf

            # TODO: attachments are not implemented for multiple robots yet
            self.inter_robots_collision_fn.append(get_inter_robots_collision_fn(
                                                self.robots, self.joints, attachments, robot_id=r,
                                                num_robots=num_robots, use_aabb=use_aabb, cache=cache,
                                                max_distance=max_distance))
        self.nodes = [OptimalNode(self.start, num_robots=self.num_robots, path=[])]

    def debug_neighbors(self):  # TODO: hard-coded for two robots
        explore_indices = [0]
        while True:
            new_explore_indices = []
            for i in explore_indices:
                print('explore_indices', explore_indices)
                print('len(self.nodes)', len(self.nodes))
                node = self.nodes[i]
                configs = get_sub_q_list(node.config, self.num_robots)
                neighbors = []
                for r in range(self.num_robots):
                    each_neighbors = []
                    for e in self.roadmaps[r].edges:
                        if e.v1.q == configs[r]: each_neighbors.append(e.v2.q)
                        if e.v2.q == configs[r]: each_neighbors.append(e.v1.q)
                    neighbors.append(each_neighbors)
                for n0 in neighbors[0]:
                    if not is_duplicate(self.nodes, n0 + configs[1]):
                        paths = []
                        paths.append(self.roadmaps[0](configs[0], n0))
                        paths.append([configs[1]])
                        # self.visualize(paths, configs)
                        if not is_local_collision(paths, self.inter_robots_collision_fn):
                            self.nodes.append(OptimalNode(n0 + configs[1], parent=get_node_from_tree(self.nodes, node.config),
                                                          num_robots=self.num_robots, path=paths))
                            new_explore_indices.append(len(self.nodes) - 1)
                            if n0 + configs[1] == self.roadmaps[0].final_conf + self.roadmaps[1].final_conf:
                                a = 1
                for n1 in neighbors[1]:
                    if not is_duplicate(self.nodes, configs[0] + n1):
                        paths = []
                        paths.append([configs[0]])
                        paths.append(self.roadmaps[1](configs[1], n1))
                        # self.visualize(paths, configs)
                        if not is_local_collision(paths, self.inter_robots_collision_fn):
                            self.nodes.append(OptimalNode(configs[0] + n1, parent=get_node_from_tree(self.nodes, node.config),
                                                          num_robots=self.num_robots, path=paths))
                            new_explore_indices.append(len(self.nodes) - 1)
                            if configs[0] + n1 == self.roadmaps[0].final_conf + self.roadmaps[1].final_conf:
                                a = 1
                for n0 in neighbors[0]:
                    for n1 in neighbors[1]:
                        if not is_duplicate(self.nodes, n0 + n1):
                            paths = []
                            paths.append(self.roadmaps[0](configs[0], n0))
                            paths.append(self.roadmaps[1](configs[1], n1))
                            # self.visualize(paths, configs)
                            if not is_local_collision(paths, self.inter_robots_collision_fn):
                                self.nodes.append(OptimalNode(n0 + n1, parent=get_node_from_tree(self.nodes, node.config),
                                                              num_robots=self.num_robots, path=paths))
                                new_explore_indices.append(len(self.nodes) - 1)
                                if n0 + n1 == self.roadmaps[0].final_conf + self.roadmaps[1].final_conf:
                                    a = 1
            explore_indices = new_explore_indices
            if not explore_indices: break
        a = 1

    def visualize(self, paths, configs):
        for r in range(self.num_robots):
            set_joint_positions(self.robots[r], self.joints[r], configs[r])
        wait_if_gui('ready')
        for r in range(self.num_robots):
            for i in range(get_max_length_list(paths)):
                if len(paths[r]) <= i:
                    set_joint_positions(self.robots[r], self.joints[r], paths[r][len(paths[r]) - 1])
                else:
                    set_joint_positions(self.robots[r], self.joints[r], paths[r][i])
            wait_for_duration(0.5)
        wait_if_gui('finish')

    def grow(self, num_iters=10, start_time=0., time_limit=math.inf, use_debug_plot=False,
             use_debug_verbal=True, debug_robot_id=0):
        best_paths = []
        best_path_cost = float('inf')
        sub_q_last = get_sub_q_list(self.start, self.num_robots)
        sub_goals = get_sub_q_list(self.goal, self.num_robots)

        while time.time() < start_time + time_limit:
            if use_debug_verbal: print('New loop starts. Time taken so far: %.2fs' % (time.time() - start_time))
            for i in range(num_iters):
                if sub_q_last is None:
                    sub_samples = []
                    for r in range(self.num_robots): sub_samples.append(self.roadmaps[r].sub_sample_fn())
                    sub_q_near = get_sub_q_near(self.nodes, self.roadmaps, sub_samples)
                else:
                    sub_samples = sub_goals
                    sub_q_near = sub_q_last

                q_new = ()
                for r in range(self.num_robots):
                    if sub_samples[r] == sub_goals[r]:
                        q_new += get_min_heuristic_vertex(self.roadmaps[r], sub_q_near[r], self.heuristic_vals[r])
                    else:
                        q_new += get_random_neighbor_vertex(self.roadmaps[r], sub_q_near[r])

                neighbor_nodes = get_neighbor_vertices(self.nodes, self.roadmaps, q_new)
                q_best, local_dist, local_paths = get_q_best(self.nodes, self.roadmaps, self.num_robots,
                                                             self.inter_robots_collision_fn, neighbor_nodes, q_new)
                if not q_best:
                    if use_debug_verbal: print('q_best is empty.')
                    sub_q_last = None
                    continue

                if best_paths:
                    if get_node_from_tree(self.nodes, q_best).cost + local_dist > best_path_cost:
                        if use_debug_verbal: print('Cost of q_new is larger than that of the best path.')
                        sub_q_last = None
                        continue

                if not is_duplicate(self.nodes, q_new):
                    if use_debug_verbal: print('New node is added to the tree.')
                    self.nodes.append(OptimalNode(q_new, num_robots=self.num_robots, d=local_dist,
                                                  parent=get_node_from_tree(self.nodes, q_best), path=local_paths))
                    if is_goal_in_tree(self.nodes, self.goal):
                        goal_node = get_node_from_tree(self.nodes, self.goal)
                        best_paths = goal_node.retrace()
                        best_path_cost = goal_node.cost
                else:
                    if use_debug_verbal: print('Rewiring: q_new is already in the tree.')
                    if q_best == q_new:
                        if use_debug_verbal: print('q_new == q_best so skip rewiring.')
                    else:
                        rewire(self.nodes, self.roadmaps, self.num_robots, q_best, q_new)

                for n in neighbor_nodes:
                    local_paths = get_local_paths(roadmap=self.roadmaps, num_robots=self.num_robots,
                                                  start_configs=get_sub_q_list(q_new, self.num_robots),
                                                  target_configs=get_sub_q_list(n, self.num_robots))
                    if get_node_from_tree(self.nodes, q_new).cost + compute_local_dist(self.roadmaps, q_new, n) < \
                        get_node_from_tree(self.nodes, n).cost and \
                        not is_local_collision(local_paths, self.inter_robots_collision_fn):
                        if use_debug_verbal: print('Rewiring: neighbor nodes.')
                        if n == q_new:
                            if use_debug_verbal: print('q_new == q_neighbor so skip rewiring.')
                        else:
                            rewire(self.nodes, self.roadmaps, self.num_robots, q_new, n)

                if get_heuristic_val(self.roadmaps, q_new, self.heuristic_vals) < \
                        get_heuristic_val(self.roadmaps, q_best, self.heuristic_vals):
                    sub_q_last = get_sub_q_list(q_new, self.num_robots)
                else:
                    sub_q_last = None
                if time.time() - start_time > time_limit: break

            # Connect to target
            if use_debug_verbal: print('Trying to connect to the target goal. ')
            sub_goals = get_sub_q_list(self.goal, self.num_robots)
            sub_q_near = get_sub_q_near(self.nodes, self.roadmaps, sub_goals)
            if sub_q_near == sub_goals:
                if use_debug_verbal: print('Goal has already reached.')
            else:
                parent_node_index = get_parent_node_index(self.nodes, sub_q_near)
                local_paths = connect_to_target(roadmap=self.roadmaps, num_robots=self.num_robots,
                                                start_configs=self.nodes[parent_node_index].sub_config,
                                                target_configs=sub_goals)
                if not local_paths:
                    if use_debug_verbal: print('Connecting to target is in collision with obstacles.')
                else:
                    local_paths = get_collision_free_paths_to_target(local_paths, self.inter_robots_collision_fn)
                    if use_debug_verbal: print('New node is added to the tree.')
                    dist_to_goal = compute_local_dist(self.roadmaps, self.nodes[parent_node_index].config, self.goal)
                    self.nodes.append(OptimalNode(self.goal, num_robots=self.num_robots, d=dist_to_goal,
                                                  parent=self.nodes[parent_node_index], path=local_paths))
                    best_paths, best_path_cost = self.update_best_path(best_paths, best_path_cost)
            if time.time() - start_time > time_limit:
                if use_debug_verbal: print('Time out.')
                break

        if best_paths: print('dRRT* is solved successfully.')
        else: SystemExit('TIMEOUT: dRRT* is NOT solved.')
        print('Spent %.2fs for dRRT*.' % (time.time()-start_time))
        return get_node_from_tree(self.nodes, self.goal).retrace()

    def update_best_path(self, best_paths, best_path_cost):
        if is_goal_in_tree(self.nodes, self.goal):
            goal_node = get_node_from_tree(self.nodes, self.goal)
            if goal_node.cost < best_path_cost:
                best_paths = goal_node.retrace()
                best_path_cost = goal_node.cost
        return best_paths, best_path_cost