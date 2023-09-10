import numpy as np
import time
import operator

from external.pybullet_planning.motion.motion_planners.utils import INF, default_selector
from external.pybullet_planning.motion.motion_planners.prm import PRM, DistancePRM


class DegreePRM(PRM):

    def __init__(self, initial_conf, final_conf, sub_sample_fn, sub_distance_fn, sub_extend_fn,
                 sub_collision_fn, samples=[], target_degree=4, connect_distance=INF,
                 attachments=[], expand_type=None, expand_configs=None):
        self.initial_conf = initial_conf
        self.final_conf = final_conf
        self.sub_sample_fn = sub_sample_fn
        self.sub_distance_fn = sub_distance_fn
        self.sub_extend_fn = sub_extend_fn
        self.sub_collision_fn = sub_collision_fn

        self.target_degree = target_degree
        self.connect_distance = connect_distance
        self.attachments = attachments
        self.expand_type = expand_type
        self.expand_configs = expand_configs
        self.expand_dim = len(expand_configs)
        super(self.__class__, self).__init__(
            sub_distance_fn, sub_extend_fn, sub_collision_fn, samples=samples)

    def grow(self, samples):
        # TODO: do sorted edges version
        new_vertices = self.add(samples)
        if self.target_degree == 0:
            return new_vertices
        if self.expand_type == 'arm':
            for v1 in new_vertices:
                degree = 0
                for _, v2 in sorted(filter(lambda pair: (pair[1] != v1) and (pair[0] <= self.connect_distance),
                                           map(lambda v: (self.distance_fn(v1.q[:-1 * self.expand_dim], v.q[:-1 * self.expand_dim]), v),
                                               self.vertices.values())),
                                    key=operator.itemgetter(0)): # TODO - slow, use nearest neighbors
                    if self.target_degree <= degree:
                        break
                    if v2 not in v1.edges:
                        path = list(self.extend_fn(v1.q[:-1 * self.expand_dim], v2.q[:-1 * self.expand_dim]))[:-1]
                        if not any(self.collision_fn(q) for q in default_selector(path)):
                            path = [p + self.expand_configs for p in path]
                            self.connect(v1, v2, path)
                            degree += 1
                    else:
                        degree += 1
                # early termination of roadmap as soon as finding a path (currently not used when planning for base poses)
                # if self(self.initial_conf, self.final_conf):
                #     return new_vertices
        elif self.expand_type == 'base':
            for v1 in new_vertices:
                degree = 0
                for _, v2 in sorted(filter(lambda pair: (pair[1] != v1) and (pair[0] <= self.connect_distance),
                                           map(lambda v: (self.distance_fn(v1.q[self.expand_dim:], v.q[self.expand_dim:]), v),
                                               self.vertices.values())),
                                    key=operator.itemgetter(0)): # TODO - slow, use nearest neighbors
                    if self.target_degree <= degree:
                        break
                    if v2 not in v1.edges:
                        path = list(self.extend_fn(v1.q[self.expand_dim:], v2.q[self.expand_dim:]))[:-1]
                        if not any(self.collision_fn(q) for q in default_selector(path)):
                            path = [self.expand_configs + p for p in path]
                            self.connect(v1, v2, path)
                            degree += 1
                    else:
                        degree += 1
                # early termination of roadmap as soon as finding a path
                if self(self.initial_conf, self.final_conf):
                    return new_vertices
        # for v1 in new_vertices:
        #     degree = 0
        #     for _, v2 in sorted(filter(lambda pair: (pair[1] != v1) and (pair[0] <= self.connect_distance),
        #                                map(lambda v: (self.distance_fn(v1.q, v.q), v), self.vertices.values())),
        #                         key=operator.itemgetter(0)): # TODO - slow, use nearest neighbors
        #         if self.target_degree <= degree:
        #             break
        #         if v2 not in v1.edges:
        #             path = list(self.extend_fn(v1.q, v2.q))[:-1]
        #             if not any(self.collision_fn(q) for q in default_selector(path)):
        #                 self.connect(v1, v2, path)
        #                 degree += 1
        #         else:
        #             degree += 1
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


def prm(start, goal, sub_distance_fn, sub_sample_fn, sub_extend_fn, sub_collision_fn,
        use_drrt_star=False, use_debug_plot=False, debug_roadmap_fn=None,
        target_degree=4, connect_distance=INF, num_samples=100, attachments=[],
        expand_type=None, expand_configs=None, use_debug=False): #, max_time=INF):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param sub_distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sub_sample_fn: Sample function - sample_fn()->conf
    :param sub_extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param sub_collision_fn: Collision function - collision_fn(q)->bool
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: compute_graph
    start_time = time.time()
    start = tuple(start)
    goal = tuple(goal)
    samples = [start, goal] + [tuple(sub_sample_fn()) for _ in range(num_samples)]
    if expand_type == 'arm':
        samples = [s + expand_configs for s in samples]
        start = start + expand_configs
        goal = goal + expand_configs
    elif expand_type == 'base':
        samples = [expand_configs + s for s in samples]
        start = expand_configs + start
        goal = expand_configs + goal
    if target_degree is None:
        roadmap = DistancePRM(sub_distance_fn, sub_extend_fn, sub_collision_fn, samples=samples,
                              connect_distance=connect_distance)
    else:
        roadmap = DegreePRM(start, goal, sub_sample_fn, sub_distance_fn, sub_extend_fn, sub_collision_fn,
                            samples=samples, target_degree=target_degree, connect_distance=connect_distance,
                            attachments=attachments, expand_type=expand_type, expand_configs=expand_configs)
    if use_debug_plot: debug_roadmap_fn(roadmap, start, goal)
    if not roadmap(start, goal):
        if use_debug:
            if expand_type == 'arm': error_type = 'base'
            elif expand_type == 'base': error_type = 'arm'
            # raise SystemExit('ERROR: Number of samples is not enough to find a path in a roadmap. Increase the sample size. '
            #                  'Problem type: {}'.format(error_type))
            print('Number of samples is not enough to find a path in a roadmap. Increase the sample size. Problem type: {}'.format(error_type))
        return None, None
    if use_drrt_star:
        heuristic_val = compute_heuristics(roadmap, sub_distance_fn)
    else:
        heuristic_val = None
    if use_debug:
        print('Spent %.2fs to find a path in a roadmap.' % (time.time()-start_time))
    return roadmap, heuristic_val