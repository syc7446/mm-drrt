# MM-dRRT

## Installation
```
git clone --recursive https://github.com/syc7446/mm-drrt.git
cd mm-drrt/
pip install -r requirements.txt
python -m main
```

## Inputs and Parameters
### Inputs
All the relevant files are included in the `/examples/envs/` folder. Find the `create_plan_order_constraints` function in the environment class, where inputs for a problem instance can be specified.
- **plan**: All the abstract actions are specified as follows. {action name (e.g. 'a0'): (robot ID, movable object ID, fixed object ID (move from), fixed object ID (move to))}
- **action_orders**: The sequence of abstract actions for each robot is specified as follows. {robot ID: (sequence of action names (e.g. 'a0', 'a1', 'a4', 'a6'))}
- **obj_orders**: The sequence of abstract actions for each movable object is specified as follows. {movable object ID: [sequence of action names (e.g. 'a1', 'a5')]}
- **init_order_constraints**: All the temporal constraints among different robots' abstract actions are specified. For example, Robot 1 must pick up Movable object 2 before Robot 2 picks up Movable object 3 from the same Fixed object, such as a table. {'pre': action name, 'post': action name}

### Parameters
- **num_robots**: The total number of robots used.
- **num_objs**: The total number of movable objects.
- **num_placement_samples**: The total number of samples used for the Place action.
- **num_base_samples**: The total number of samples used for solving a base motion planning problem. For example, if PRM is used, this parameter specifies the total number of samples used to construct the roadmap.
- **num_arm_samples**: This parameter is equivalent to **num_base_samples** but for arm motions.
- **env_type**: This parameter determines which example to run.
- **use_gui**: This parameter determines whether to use a GUI for visualization.

## Examples
We provide two example codes to facilitate understanding of how MM-dRRT works. The first example involves a single robot picking up an object from one table and placing it on another. The second example involves two robots moving two objects located on two tables.

- Single robot, single object: Set both `num_robots` and `num_objs` to 1. Set `env_type` to `exp_single_robot`. 
- Two robots, two objects: Set both `num_robots` and `num_objs` to 2. Set `env_type` to `exp_two_robots`. 

In both examples, focus on the `_create_problem` function in `example_single_robot_env` and `example_two_robots_env` files located in the `/examples/envs/` folder to understand how the environments are specified.