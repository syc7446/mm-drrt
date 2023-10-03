import numpy as np
import networkx as nx
import time

from mm_drrt.utils.task_planner_utils import initialize_actions, initialize_robot_plans, subgoal_refinement, \
    individual_path_computation
from mm_drrt.utils.motion_planner_utils import get_max_length_list, get_arm_positions, base_motion


class Robot:
    def __init__(self, id):
        self.id = id
        self.actions = []

    def __repr__(self):
        return 'robot_' + self.id


class PlanSkeleton:
    def __init__(self, env, plan, obj_orders, init_order_constraints, num_samples, use_debug=False):
        self.env = env
        self.robot_plans = {}
        self.actions = {}
        self.obj_orders = obj_orders
        self.f_objs = {}
        self.num_samples = num_samples
        self.use_debug = use_debug

        self.initialize(plan, init_order_constraints)

    '''
    Initilize Action and Fixedobj classes
    '''
    def initialize(self, plan, init_order_constraints):
        saved_world = self.env.save_world()
        initialize_actions(plan, self.env, self.actions, init_order_constraints, self.f_objs, self.num_samples)
        initialize_robot_plans(plan, self.env, self.actions, self.robot_plans)

        for f_obj in self.f_objs:
            self.f_objs[f_obj].set_obj_orders(self.obj_orders)
        self.env.restore_world(saved_world)


    def plan_refinement(self, num_base_samples, num_arm_samples, drrt_num_iters, drrt_time_limit):
        # TODO: currently, backtracking is performed brute-force. we could use the failure information to find more promising values in the early stage (informed search)
        while True:
            '''
            Step 1: placement refinement
            '''
            start = time.time()
            for f in self.f_objs:
                while True:
                    if self.f_objs[f].is_refine_placements():
                        break
            print('Step 1: placement refinement succeeded')
            print('        time taken ', time.time() - start)

            '''
            Step 2: subgoal refinement
            '''
            start = time.time()
            if not subgoal_refinement(self.robot_plans, self.env, self.obj_orders, self.actions, self.use_debug):
                continue
            print('Step 2: subgoal refinement succeeded')
            print('        time taken ', time.time() - start)

            '''
            Step 3: individual path computation
            '''
            start = time.time()
            roadmaps, heuristic_vals = [], []
            result, roadmaps, heuristic_vals =  individual_path_computation(self.robot_plans, self.env,
                                                                            num_base_samples, num_arm_samples,
                                                                            roadmaps, heuristic_vals, self.use_debug)
            if not result: continue
            print('Step 3: individual path computation succeeded')
            print('        time taken ', time.time() - start)

            '''
            Step 4: composite path computation
            '''
            # hard-coded sequences
            paths, attachments = [], []

            path = []
            path_0 = roadmaps[0][0](roadmaps[0][0].initial_conf, roadmaps[0][0].final_conf)
            for i in range(len(path_0)):
                q = ()
                q += path_0[i]
                q += roadmaps[1][0].initial_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_0 = roadmaps[0][1](roadmaps[0][1].initial_conf, roadmaps[0][1].final_conf)
            for i in range(len(path_0)):
                q = ()
                q += path_0[i]
                q += roadmaps[1][0].initial_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_0 = roadmaps[0][2](roadmaps[0][2].initial_conf, roadmaps[0][2].final_conf)
            for i in range(len(path_0)):
                q = ()
                q += path_0[i]
                q += roadmaps[1][0].initial_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][2].attachments[0], None, None])

            path = []
            path_0 = roadmaps[0][3](roadmaps[0][3].initial_conf, roadmaps[0][3].final_conf)
            for i in range(len(path_0)):
                q = ()
                q += path_0[i]
                q += roadmaps[1][0].initial_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][3].attachments[0], None, None])

            path = []
            path_0 = roadmaps[0][4](roadmaps[0][4].initial_conf, roadmaps[0][4].final_conf)
            for i in range(len(path_0)):
                q = ()
                q += path_0[i]
                q += roadmaps[1][0].initial_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][4].attachments[0], None, None])

            path = []
            path_0 = roadmaps[0][5](roadmaps[0][5].initial_conf, roadmaps[0][5].final_conf)
            for i in range(len(path_0)):
                q = ()
                q += path_0[i]
                q += roadmaps[1][0].initial_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_0 = roadmaps[0][6](roadmaps[0][6].initial_conf, roadmaps[0][6].final_conf)
            path_1 = roadmaps[1][0](roadmaps[1][0].initial_conf, roadmaps[1][0].final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][6].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][0].final_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_0 = roadmaps[0][7](roadmaps[0][7].initial_conf, roadmaps[0][7].final_conf)
            path_1 = roadmaps[1][1](roadmaps[1][1].initial_conf, roadmaps[1][1].final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][7].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][1].final_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_0 = roadmaps[0][8](roadmaps[0][8].initial_conf, roadmaps[0][8].final_conf)
            path_1 = roadmaps[1][2](roadmaps[1][2].initial_conf, roadmaps[1][2].final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][8].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][2].final_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][8].attachments[0], roadmaps[1][2].attachments[0], None])

            path = []
            path_0 = roadmaps[0][9](roadmaps[0][9].initial_conf, roadmaps[0][9].final_conf)
            path_1 = roadmaps[1][3](roadmaps[1][3].initial_conf, roadmaps[1][3].final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][9].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][3].final_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][9].attachments[0], roadmaps[1][3].attachments[0], None])

            path = []
            path_0 = roadmaps[0][10](roadmaps[0][10].initial_conf, roadmaps[0][10].final_conf)
            path_1 = roadmaps[1][4](roadmaps[1][4].initial_conf, roadmaps[1][4].final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][10].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][4].final_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][10].attachments[0], roadmaps[1][4].attachments[0], None])

            path = []
            path_0 = roadmaps[0][11](roadmaps[0][11].initial_conf, roadmaps[0][11].final_conf)
            path_1 = roadmaps[1][5](roadmaps[1][5].initial_conf, roadmaps[1][5].final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][11].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][5].final_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_0 = roadmaps[0][12](roadmaps[0][12].initial_conf, roadmaps[0][12].final_conf)
            for i in range(len(path_0)):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][12].final_conf
                q += roadmaps[1][5].final_conf
                q += roadmaps[2][0].initial_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_2 = roadmaps[2][0](roadmaps[2][0].initial_conf, roadmaps[2][0].final_conf)
            for i in range(len(path_2)):
                q = ()
                q += roadmaps[0][12].final_conf
                q += roadmaps[1][5].final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][0].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_0 = roadmaps[0][13](roadmaps[0][13].initial_conf, roadmaps[0][13].final_conf)
            path_2 = roadmaps[2][1](roadmaps[2][1].initial_conf, roadmaps[2][1].final_conf)
            for i in range(get_max_length_list([path_0, path_2])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][13].final_conf
                q += roadmaps[1][5].final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][1].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_0 = roadmaps[0][14](roadmaps[0][14].initial_conf, roadmaps[0][14].final_conf)
            path_2 = roadmaps[2][2](roadmaps[2][2].initial_conf, roadmaps[2][2].final_conf)
            for i in range(get_max_length_list([path_0, path_2])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][14].final_conf
                q += roadmaps[1][5].final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][2].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][14].attachments[0], roadmaps[2][2].attachments[0], None])

            robot = 24
            r_expand_configs = get_arm_positions(robot=robot, arm=self.env._arm)
            base_roadmap, base_heuristic_val = base_motion(robot, roadmaps[1][5].final_conf[:3],
                                                           (-2.0, 0.0, 0.0),
                                                           obstacles=self.env.fixed_obstacles, custom_limits=self.env.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=None)

            path = []
            path_0 = roadmaps[0][15](roadmaps[0][15].initial_conf, roadmaps[0][15].final_conf)
            path_1 = base_roadmap(base_roadmap.initial_conf, base_roadmap.final_conf)
            path_2 = roadmaps[2][3](roadmaps[2][3].initial_conf, roadmaps[2][3].final_conf)
            for i in range(get_max_length_list([path_0, path_1, path_2])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][15].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += base_roadmap.final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][3].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][15].attachments[0], None, roadmaps[2][3].attachments[0]])

            path = []
            path_0 = roadmaps[0][16](roadmaps[0][16].initial_conf, roadmaps[0][16].final_conf)
            path_2 = roadmaps[2][4](roadmaps[2][4].initial_conf, roadmaps[2][4].final_conf)
            for i in range(get_max_length_list([path_0, path_2])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][16].final_conf
                q += base_roadmap.final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][4].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][16].attachments[0], None, roadmaps[2][4].attachments[0]])

            path = []
            path_0 = roadmaps[0][17](roadmaps[0][17].initial_conf, roadmaps[0][17].final_conf)
            path_2 = roadmaps[2][5](roadmaps[2][5].initial_conf, roadmaps[2][5].final_conf)
            for i in range(get_max_length_list([path_0, path_2])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][17].final_conf
                q += base_roadmap.final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][5].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            robot = 24
            r_expand_configs = get_arm_positions(robot=robot, arm=self.env._arm)
            base_roadmap, base_heuristic_val = base_motion(robot, (-2.0, 0.0, 0.0),
                                                           roadmaps[1][6].final_conf[:3],
                                                           obstacles=self.env.fixed_obstacles,
                                                           custom_limits=self.env.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=None)

            path = []
            path_0 = roadmaps[0][18](roadmaps[0][18].initial_conf, roadmaps[0][18].final_conf)
            path_1 = base_roadmap(base_roadmap.initial_conf, base_roadmap.final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][18].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += base_roadmap.final_conf
                q += roadmaps[2][5].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            robot = 25
            r_expand_configs = get_arm_positions(robot=robot, arm=self.env._arm)
            base_roadmap, base_heuristic_val = base_motion(robot, roadmaps[2][5].final_conf[:3],
                                                           (2.0, -1.0, 0.0),
                                                           obstacles=self.env.fixed_obstacles,
                                                           custom_limits=self.env.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=None)

            path = []
            path_0 = roadmaps[0][19](roadmaps[0][19].initial_conf, roadmaps[0][19].final_conf)
            path_1 = roadmaps[1][7](roadmaps[1][7].initial_conf, roadmaps[1][7].final_conf)
            path_2 = base_roadmap(base_roadmap.initial_conf, base_roadmap.final_conf)
            for i in range(get_max_length_list([path_0, path_1, path_2])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][19].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][7].final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += base_roadmap.final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_0 = roadmaps[0][20](roadmaps[0][20].initial_conf, roadmaps[0][20].final_conf)
            path_1 = roadmaps[1][8](roadmaps[1][8].initial_conf, roadmaps[1][8].final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][20].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][8].final_conf
                q += base_roadmap.final_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][20].attachments[0], roadmaps[1][8].attachments[0], None])

            path = []
            path_1 = roadmaps[1][9](roadmaps[1][9].initial_conf, roadmaps[1][9].final_conf)
            for i in range(len(path_1)):
                q = ()
                q += roadmaps[0][20].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][9].final_conf
                q += base_roadmap.final_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][20].attachments[0], roadmaps[1][9].attachments[0], None])

            path = []
            path_0 = roadmaps[0][21](roadmaps[0][21].initial_conf, roadmaps[0][21].final_conf)
            path_1 = roadmaps[1][10](roadmaps[1][10].initial_conf, roadmaps[1][10].final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][21].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][10].final_conf
                q += base_roadmap.final_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][21].attachments[0], roadmaps[1][10].attachments[0], None])

            path = []
            path_0 = roadmaps[0][22](roadmaps[0][22].initial_conf, roadmaps[0][22].final_conf)
            path_1 = roadmaps[1][11](roadmaps[1][11].initial_conf, roadmaps[1][11].final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][22].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += roadmaps[1][11].final_conf
                q += base_roadmap.final_conf
                path.append(q)
            paths.append(path)
            attachments.append([roadmaps[0][22].attachments[0], None, None])
            robot_25_final_conf = base_roadmap.final_conf

            robot = 24
            r_expand_configs = get_arm_positions(robot=robot, arm=self.env._arm)
            base_roadmap, base_heuristic_val = base_motion(robot, roadmaps[1][11].final_conf[:3],
                                                           (-2.0, 0.0, 0.0),
                                                           obstacles=self.env.fixed_obstacles,
                                                           custom_limits=self.env.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=None)

            path = []
            path_0 = roadmaps[0][23](roadmaps[0][23].initial_conf, roadmaps[0][23].final_conf)
            path_1 = base_roadmap(base_roadmap.initial_conf, base_roadmap.final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += roadmaps[0][23].final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += base_roadmap.final_conf
                q += robot_25_final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])
            robot_24_final_conf = base_roadmap.final_conf

            robot = 23
            r_expand_configs = get_arm_positions(robot=robot, arm=self.env._arm)
            base_roadmap, base_heuristic_val = base_motion(robot, roadmaps[0][23].final_conf[:3],
                                                           (-6.0, -2.5, 0.0),
                                                           obstacles=self.env.fixed_obstacles,
                                                           custom_limits=self.env.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=None)

            path = []
            path_0 = base_roadmap(base_roadmap.initial_conf, base_roadmap.final_conf)
            for i in range(len(path_0)):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += base_roadmap.final_conf
                q += robot_24_final_conf
                q += robot_25_final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])
            robot_23_final_conf = base_roadmap.final_conf

            robot = 25
            r_expand_configs = get_arm_positions(robot=robot, arm=self.env._arm)
            base_roadmap, base_heuristic_val = base_motion(robot, (2.0, -1.0, 0.0),
                                                           roadmaps[2][6].final_conf[:3],
                                                           obstacles=self.env.fixed_obstacles,
                                                           custom_limits=self.env.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=None)

            path = []
            path_2 = base_roadmap(base_roadmap.initial_conf, base_roadmap.final_conf)
            for i in range(len(path_2)):
                q = ()
                q += robot_23_final_conf
                q += robot_24_final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += base_roadmap.final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_2 = roadmaps[2][7](roadmaps[2][7].initial_conf, roadmaps[2][7].final_conf)
            for i in range(len(path_2)):
                q = ()
                q += robot_23_final_conf
                q += robot_24_final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][7].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_2 = roadmaps[2][8](roadmaps[2][8].initial_conf, roadmaps[2][8].final_conf)
            for i in range(len(path_2)):
                q = ()
                q += robot_23_final_conf
                q += robot_24_final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][8].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, roadmaps[2][8].attachments[0]])

            path = []
            path_2 = roadmaps[2][9](roadmaps[2][9].initial_conf, roadmaps[2][9].final_conf)
            for i in range(len(path_2)):
                q = ()
                q += robot_23_final_conf
                q += robot_24_final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][9].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, roadmaps[2][9].attachments[0]])

            path = []
            path_2 = roadmaps[2][10](roadmaps[2][10].initial_conf, roadmaps[2][10].final_conf)
            for i in range(len(path_2)):
                q = ()
                q += robot_23_final_conf
                q += robot_24_final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][10].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, roadmaps[2][10].attachments[0]])

            path = []
            path_2 = roadmaps[2][11](roadmaps[2][11].initial_conf, roadmaps[2][11].final_conf)
            for i in range(len(path_2)):
                q = ()
                q += robot_23_final_conf
                q += robot_24_final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][11].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            path = []
            path_2 = roadmaps[2][12](roadmaps[2][12].initial_conf, roadmaps[2][12].final_conf)
            for i in range(len(path_2)):
                q = ()
                q += robot_23_final_conf
                q += robot_24_final_conf
                if i < len(path_2):
                    q += path_2[i]
                else:
                    q += roadmaps[2][12].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            robot = 23
            r_expand_configs = get_arm_positions(robot=robot, arm=self.env._arm)
            robot_23_base_roadmap, robot_23_base_heuristic_val = base_motion(robot, (-6.0, -2.5, 0.0),
                                                           roadmaps[0][24].final_conf[:3],
                                                           obstacles=self.env.fixed_obstacles,
                                                           custom_limits=self.env.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=None)

            robot = 24
            r_expand_configs = get_arm_positions(robot=robot, arm=self.env._arm)
            robot_24_base_roadmap, robot_24_base_heuristic_val = base_motion(robot, (-2.0, 0.0, 0.0),
                                                           roadmaps[1][12].final_conf[:3],
                                                           obstacles=self.env.fixed_obstacles,
                                                           custom_limits=self.env.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=None)

            path = []
            path_0 = robot_23_base_roadmap(robot_23_base_roadmap.initial_conf, robot_23_base_roadmap.final_conf)
            path_1 = robot_24_base_roadmap(robot_24_base_roadmap.initial_conf, robot_24_base_roadmap.final_conf)
            for i in range(get_max_length_list([path_0, path_1])):
                q = ()
                if i < len(path_0):
                    q += path_0[i]
                else:
                    q += robot_23_base_roadmap.final_conf
                if i < len(path_1):
                    q += path_1[i]
                else:
                    q += robot_24_base_roadmap.final_conf
                q += roadmaps[2][12].final_conf
                path.append(q)
            paths.append(path)
            attachments.append([None, None, None])

            return paths, attachments