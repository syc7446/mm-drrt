from external.pybullet_planning.pybullet_tools.utils import BLUE, RED, GREEN, BLACK, YELLOW

MODE = 'solving' # [solving, visualization]
USE_GUI = True
USE_DEBUG_PLOT = False
USE_DEBUG_VERBAL = True
USE_DRRT_STAR = True

# NUM_ROBOTS = 3
# ROBOT_COLORS = [RED, BLUE, GREEN]
NUM_ROBOTS = 2
ROBOT_COLORS = [RED, BLUE]

NUM_SAMPLES_PRM = 50
NUM_DRRT_ITERS = 10
TIME_LIMIT = 30000
DEBUG_ROBOT_ID = 0
SIZE_X = 0.50
SIZE_Y = 1.25
SLEEP = 0.05 # None | 0.05
SEED = 1

CUSTOM_LIMITS = {
    'x': (-SIZE_X, SIZE_X),
    'y': (-SIZE_Y, SIZE_Y),
}