import numpy as np
import networkx as nx
import time

from mm_drrt.utils.task_planner_utils import initialize_actions, initialize_robot_plans, subgoal_refinement, \
    individual_path_computation, composite_path_computation, assign_order_constraints


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
            start = time.time()
            robots = [r for r in self.robot_plans.keys()]
            order_constraints = assign_order_constraints(self.robot_plans)
            composite_path = composite_path_computation(self.env, robots, roadmaps, heuristic_vals, order_constraints, drrt_num_iters, drrt_time_limit)
            if composite_path:
                print('Step 4: composite path computation succeeded')
                print('        time taken ', time.time() - start)
                return composite_path
