import numpy as np
import math
import time

from legacy.evaluation.motion_planner_utils import OptimalNode, get_inter_robots_collision_fn, get_angle, \
    is_duplicate, get_sub_nodes, get_parent_node_index, get_local_paths, is_local_collision, \
    get_sub_q_near, connect_to_target, get_collision_free_paths_to_target, get_sub_q_list, \
    get_min_heuristic_vertex, get_random_neighbor_vertex, get_neighbor_vertices, compute_local_dist, \
    get_q_best, get_node_from_tree, rewire, get_heuristic_val, is_goal_in_tree, get_subprob, \
    get_substarts_subgoals, update_subprob_id, is_q_new_in_subgoal, get_goals, debug_2d_sub_sampling

from external.pybullet_planning.pybullet_tools.utils import MAX_DISTANCE

class dRRTStar:

    def __init__(self, robots, joints, roadmaps, num_robots=1, heuristic_vals=[], radius=None,
                 attachments=[], max_distance=MAX_DISTANCE, use_aabb=False, cache=True):
        self.robots = robots
        self.joints = joints
        self.joint_dim = len(self.joints[0])
        self.roadmaps = roadmaps
        self.num_robots = num_robots
        self.heuristic_vals = heuristic_vals
        self.radius = radius
        self.inter_robots_collision_fn = []

        self.starts = []
        self.goals = []
        self.subprob_id = []
        for r in range(self.num_robots):
            starts = ()
            goals = ()
            for roadmap in self.roadmaps[r]:
                starts += roadmap.initial_conf
                goals += roadmap.final_conf
            self.starts.append(starts)
            self.goals.append(goals)
            self.subprob_id.append(0)

            # TODO: attachments are not implemented for multiple robots yet
            self.inter_robots_collision_fn.append(get_inter_robots_collision_fn(
                self.robots, self.joints, attachments, robot_id=r, num_robots=num_robots,
                use_aabb=use_aabb, cache=cache, max_distance=max_distance))
        self.nodes = [OptimalNode(get_substarts_subgoals(self.starts, self.subprob_id, self.joint_dim),
                                  num_robots=self.num_robots, subprob_id=self.subprob_id, path=[])]

    def grow(self, num_iters=10, start_time=0., time_limit=math.inf, use_debug_plot=False,
             use_debug_verbal=False, debug_robot_id=0):
        best_paths = []
        best_path_cost = float('inf')
        sub_q_last = get_sub_q_list(get_substarts_subgoals(self.starts, self.subprob_id, self.joint_dim), self.num_robots)
        sub_goals = get_sub_q_list(get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim), self.num_robots)
        is_found_path = False

        while time.time() < start_time + time_limit:
            if is_found_path: break
            if use_debug_verbal: print('New loop starts. Time taken so far: %.2fs' % (time.time() - start_time))
            for i in range(num_iters):
                if sub_q_last is None:
                    sub_samples = []
                    for r in range(self.num_robots):
                        sub_samples.append(get_subprob(self.roadmaps, self.subprob_id)[r].sub_sample_fn())
                    sub_q_near = get_sub_q_near(self.nodes, get_subprob(self.roadmaps, self.subprob_id), sub_samples, self.subprob_id)
                else:
                    sub_samples = sub_goals
                    sub_q_near = sub_q_last

                q_new = ()
                for r in range(self.num_robots):
                    if sub_samples[r] == sub_goals[r]:
                        q_new += get_min_heuristic_vertex(get_subprob(self.roadmaps, self.subprob_id)[r],
                                                          sub_q_near[r], get_subprob(self.heuristic_vals, self.subprob_id)[r])
                    else:
                        q_new += get_random_neighbor_vertex(get_subprob(self.roadmaps, self.subprob_id)[r], sub_q_near[r])

                neighbor_nodes = get_neighbor_vertices(self.nodes, get_subprob(self.roadmaps, self.subprob_id), q_new, self.subprob_id)
                q_best, local_dist, local_paths = get_q_best(self.nodes, get_subprob(self.roadmaps, self.subprob_id), self.num_robots,
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

                is_update_subprob_id = False
                if not is_duplicate(self.nodes, q_new, self.subprob_id):
                    if is_q_new_in_subgoal(get_subprob(self.roadmaps, self.subprob_id), q_new,
                                           get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim)):
                        if use_debug_verbal: print('Some of robots reached their subgoals.')
                        sub_q_last, sub_goals = update_subprob_id(q_new, sub_goals, self.subprob_id,
                                                                  self.goals, self.joint_dim, self.roadmaps)
                        is_update_subprob_id = True

                    if use_debug_verbal: print('New node is added to the tree.')
                    self.nodes.append(OptimalNode(q_new, num_robots=self.num_robots, d=local_dist,
                                                  parent=get_node_from_tree(self.nodes, q_best),
                                                  subprob_id=self.subprob_id, path=local_paths))

                    # TODO: currently best path can be computed only when the final goals are reached. we do not consider optimality yet.
                    if is_goal_in_tree(self.nodes, get_goals(self.goals, self.joint_dim), self.roadmaps, self.subprob_id):
                        goal_node = get_node_from_tree(self.nodes, get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim))
                        best_paths = goal_node.retrace()
                        best_path_cost = goal_node.cost
                        is_found_path = True
                else:
                    if use_debug_verbal: print('Rewiring: q_new is already in the tree.')
                    if q_best == q_new:
                        if use_debug_verbal: print('q_new == q_best so skip rewiring.')
                    # else:
                    #     rewire(self.nodes, get_subprob(self.roadmaps, self.subprob_id), self.num_robots, q_best, q_new)

                # for n in neighbor_nodes:
                #     local_paths = get_local_paths(roadmap=get_subprob(self.roadmaps, self.subprob_id), num_robots=self.num_robots,
                #                                   start_configs=get_sub_q_list(q_new, self.num_robots),
                #                                   target_configs=get_sub_q_list(n, self.num_robots))
                #     if get_node_from_tree(self.nodes, q_new).cost + compute_local_dist(get_subprob(self.roadmaps, self.subprob_id), q_new, n) < \
                #         get_node_from_tree(self.nodes, n).cost and \
                #         not is_local_collision(local_paths, self.inter_robots_collision_fn):
                #         if use_debug_verbal: print('Rewiring: neighbor nodes.')
                #         if n == q_new:
                #             if use_debug_verbal: print('q_new == q_neighbor so skip rewiring.')
                #         else:
                #             rewire(self.nodes, get_subprob(self.roadmaps, self.subprob_id), self.num_robots, q_new, n)

                if not is_update_subprob_id:
                    if get_heuristic_val(get_subprob(self.roadmaps, self.subprob_id), q_new,
                                         get_subprob(self.heuristic_vals, self.subprob_id)) < \
                            get_heuristic_val(get_subprob(self.roadmaps, self.subprob_id), q_best,
                                              get_subprob(self.heuristic_vals, self.subprob_id)):
                        sub_q_last = get_sub_q_list(q_new, self.num_robots)
                    else:
                        sub_q_last = None
                if time.time() - start_time > time_limit: break

            # Connect to target
            if use_debug_verbal: print('Trying to connect to the target goal. ')
            sub_goals = get_sub_q_list(get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim), self.num_robots)
            sub_q_near = get_sub_q_near(self.nodes, get_subprob(self.roadmaps, self.subprob_id), sub_goals, self.subprob_id)
            if sub_q_near == sub_goals:
                if use_debug_verbal: print('Goal has already reached.')
            else:
                parent_node_index = get_parent_node_index(self.nodes, sub_q_near)
                # in connect_to_target, collisions with obstacles are only checked. inter-robot collisions are checked only for starts and goals
                local_paths = connect_to_target(roadmap=get_subprob(self.roadmaps, self.subprob_id),
                                                num_robots=self.num_robots,
                                                start_configs=self.nodes[parent_node_index].sub_config,
                                                target_configs=sub_goals,
                                                drrt_collision_fn=self.inter_robots_collision_fn)
                if not local_paths:
                    if use_debug_verbal: print('Connecting to target is in collision with obstacles.')
                else:
                    # in get_collision_free_paths_to_target, collisions with other robots are checked
                    local_paths = get_collision_free_paths_to_target(local_paths, self.inter_robots_collision_fn)
                    if use_debug_verbal: print('New node is added to the tree.')
                    dist_to_goal = compute_local_dist(get_subprob(self.roadmaps, self.subprob_id),
                                                      self.nodes[parent_node_index].config,
                                                      get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim))
                    node_pose = get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim)
                    sub_q_last, sub_goals = update_subprob_id(get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim),
                                                              sub_goals, self.subprob_id, self.goals, self.joint_dim, self.roadmaps)
                    self.nodes.append(OptimalNode(node_pose, num_robots=self.num_robots, d=dist_to_goal,
                                                  parent=self.nodes[parent_node_index],
                                                  subprob_id=self.subprob_id, path=local_paths))

                    # TODO: currently best path can be computed only when the final goals are reached. we do not consider optimality yet.
                    if is_goal_in_tree(self.nodes, get_goals(self.goals, self.joint_dim), self.roadmaps, self.subprob_id):
                        goal_node = get_node_from_tree(self.nodes, get_substarts_subgoals(self.goals, self.subprob_id,
                                                                                          self.joint_dim))
                        best_paths = goal_node.retrace()
                        best_path_cost = goal_node.cost
                        is_found_path = True
                    # best_paths, best_path_cost = self.update_best_path(best_paths, best_path_cost)
            if time.time() - start_time > time_limit:
                if use_debug_verbal: print('Time out.')
                break

        if best_paths: print('dRRT* is solved successfully.')
        else: SystemExit('TIMEOUT: dRRT* is NOT solved.')
        print('Spent %.2fs for dRRT*.' % (time.time()-start_time))
        return get_node_from_tree(self.nodes, get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim)).retrace()

    def update_best_path(self, best_paths, best_path_cost):
        if is_goal_in_tree(self.nodes, get_goals(self.goals, self.joint_dim), self.roadmaps, self.subprob_id): # TODO: currently best path can be computed only when the final goals are reached.
            goal_node = get_node_from_tree(self.nodes, get_substarts_subgoals(self.goals, self.subprob_id, self.joint_dim))
            if goal_node.cost < best_path_cost:
                best_paths = goal_node.retrace()
                best_path_cost = goal_node.cost
        return best_paths, best_path_cost
