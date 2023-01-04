import numpy as np
import time
import operator

from external.pybullet_planning.motion.motion_planners.utils import INF, default_selector
from external.pybullet_planning.motion.motion_planners.prm import DistancePRM, PRM


class DegreePRM(PRM):

    def __init__(self, initial_conf, final_conf, sample_fn, distance_fn, extend_fn,
                 collision_fn, samples=[], target_degree=4, connect_distance=INF):
        self.initial_conf = initial_conf
        self.final_conf = final_conf

        self.target_degree = target_degree
        self.connect_distance = connect_distance
        super(self.__class__, self).__init__(
            distance_fn, extend_fn, collision_fn, samples=samples)

    def grow(self, samples):
        # TODO: do sorted edges version
        new_vertices = self.add(samples)
        if self.target_degree == 0:
            return new_vertices
        for v1 in new_vertices:
            degree = 0
            for _, v2 in sorted(filter(lambda pair: (pair[1] != v1) and (pair[0] <= self.connect_distance),
                                       map(lambda v: (self.distance_fn(v1.q, v.q), v), self.vertices.values())),
                                key=operator.itemgetter(0)): # TODO - slow, use nearest neighbors
                if self.target_degree <= degree:
                    break
                if v2 not in v1.edges:
                    path = list(self.extend_fn(v1.q, v2.q))[:-1]
                    if not any(self.collision_fn(q) for q in default_selector(path)):
                        self.connect(v1, v2, path)
                        degree += 1
                else:
                    degree += 1
        return new_vertices


def get_vertex_index(list, val):
    for i in range(len(list)):
        if list[i][1] == val:
            return i
    raise SystemExit('ERROR: No vertex exists in the list.')


def compute_heuristics(roadmap, distance_fn):
    graph_size = len(roadmap.values())
    graph = np.zeros((graph_size, graph_size))
    vertices_list = list(roadmap.vertices.items())
    for e in roadmap.edges:
        graph[get_vertex_index(vertices_list, e.v1),
              get_vertex_index(vertices_list, e.v2)] = distance_fn(e.v1.q, e.v2.q)
        graph[get_vertex_index(vertices_list, e.v2),
              get_vertex_index(vertices_list, e.v1)] = distance_fn(e.v2.q, e.v1.q)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import johnson
    dist_matrix = johnson(csgraph=csr_matrix(graph.tolist()))
    heuristic_val = {}
    for i in range(graph_size):
        heuristic_val[vertices_list[i][1]] = dist_matrix[i][1] # dist_matrix[i][1] == goal config
    return heuristic_val


def prm(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
        use_drrt_star=False, use_debug_plot=False, debug_roadmap_fn=None,
        target_degree=4, connect_distance=INF, num_samples=100): #, max_time=INF):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: compute_graph
    start_time = time.time()
    start = tuple(start)
    goal = tuple(goal)
    samples = [start, goal] + [tuple(sample_fn()) for _ in range(num_samples)]
    if target_degree is None:
        roadmap = DistancePRM(distance_fn, extend_fn, collision_fn, samples=samples,
                              connect_distance=connect_distance)
    else:
        roadmap = DegreePRM(start, goal, sample_fn, distance_fn, extend_fn, collision_fn,
                            samples=samples, target_degree=target_degree, connect_distance=connect_distance)
    if use_debug_plot: debug_roadmap_fn(roadmap, start, goal)
    if not roadmap(start, goal):
        raise SystemExit('ERROR: Number of samples is not enough to find a path in a roadmap. Increase the sample size.')
    if use_drrt_star:
        heuristic_val = compute_heuristics(roadmap, distance_fn)
    else:
        heuristic_val = None
    print('Spent %.2fs to find a path in a roadmap.' % (time.time()-start_time))
    return roadmap, heuristic_val