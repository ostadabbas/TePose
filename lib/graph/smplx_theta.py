import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from lib.graph import tools

num_node = 24
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 4), (4, 7), (10, 7), (13, 10), (16, 13), (14, 10), (17, 14),
                    (19, 17), (21, 19), (23, 21), (15, 10), (18, 15), (20, 18),
                    (22, 20), (24, 22), (2, 1), (5, 2), (8, 5), (11, 8),
                    (3, 1), (6, 3), (9, 6), (12, 9)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
