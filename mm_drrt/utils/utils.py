import numpy as np
from itertools import islice

from external.pybullet_planning.pybullet_tools.ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics
from external.pybullet_planning.pybullet_tools.pr2_utils import get_gripper_link, get_arm_joints, arm_conf, open_arm, \
    get_aabb, get_disabled_collisions, get_group_joints, learned_pose_generator, PR2_GROUPS
from external.pybullet_planning.pybullet_tools.pr2_primitives import create_trajectory, iterate_approach_path, Commands, \
    State, SELF_COLLISIONS, Pose, Conf
from external.pybullet_planning.pybullet_tools.utils import is_placement, multiply, invert, set_joint_positions, \
    pairwise_collision, get_sample_fn, get_distance_fn, check_initial_end, \
    get_joint_positions, plan_direct_joint_motion, joint_from_name, all_between, BodySaver, \
    LockRenderer, get_bodies, get_joint_limits, set_joint_limits, get_default_resolution, uniform_pose_generator, Saver, \
    PoseSaver, ConfSaver, get_configuration, remove_body, inverse_kinematics_helper, get_movable_joints, get_link_pose, \
    is_pose_close, elapsed_time, irange, create_sub_robot, get_custom_limits, sub_inverse_kinematics, INF, \
    get_box_geometry, create_shape, create_body, sample_placement, get_pose, get_euler, STATIC_MASS, RED, BROWN, \
    join_paths, get_parent_dir, get_extend_fn, get_collision_fn, MAX_DISTANCE

from mm_drrt.planner.prm import prm


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


def base_motion(robot, base_start, base_goal, teleport=False, obstacles=[], attachments=[], custom_limits={}):
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
            base_path = plan_joint_motion(robot, base_joints, base_goal, obstacles=obstacles,
                                          attachments=attachments, disabled_collisions=disabled_collisions,
                                          resolutions=resolutions, custom_limits=custom_limits)
        if not base_path: set_joint_positions(robot, base_joints, base_start)
        return base_path


def get_ir_sampler(problem, custom_limits={}, max_attempts=25, collisions=True, collision_objs=[], learned=True):
    robot = problem.robot
    obstacles = collision_objs if collisions else []
    gripper = problem.get_gripper()

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


def get_ik_fn(problem, custom_limits={}, collisions=True, collision_objs=[], teleport=False):
    robot = problem.robot
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
        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.05 ** np.ones(len(arm_joints))
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions)
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
        return (cmd, attachments, gripper_pose[0],)  # Only this line has been changed from the original code

    return fn


def get_ik_ir_gen(problem, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)

    def gen(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
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