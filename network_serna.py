import numpy as np


class Node(object):
    def __init__(self, id):
        self.id = id
        self.activated = False
        self.neighbors = []
        self.degree = 0

    def add_neighborg(self, node):
        self.neighbors.append(node)
        self.degree += 1

    def get_id(self):
        return self.id

    def get_neighborgs(self):
        return self.neighbors

    def is_active(self):
        return self.activated

    def set_inactive(self):
        self.activated = False

    def set_active(self):
        self.activated = True

    def get_degree(self):
        return self.degree

    def __str__(self):
        temp = "ID: " + str(self.id)
        return temp + "\n"


class Graph(object):
    def __init__(self, n_nodes, connectivity):

        self.n_nodes = n_nodes
        self.connectivity = connectivity

        self.adj_matrix = np.zeros([n_nodes, n_nodes], dtype=np.float)
        self.nodes = [Node(id) for id in range(self.n_nodes)]

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if np.random.rand() <= self.connectivity:
                    self.nodes[i].add_neighborg(self.nodes[j])
                    self.adj_matrix[i][j] = np.random.rand()

    def get_nodes(self):
        return self.nodes

    def monte_carlo_sampling(self, seeds, max_repetition):

        nodes_activ_prob = np.zeros(self.n_nodes, dtype=np.float)
        for _ in range(max_repetition):
            for node in self.nodes:
                if node in seeds:
                    node.set_active()
                else:
                    node.set_inactive()

            live_edges = self.adj_matrix > np.random.rand(self.n_nodes, self.n_nodes)
            new_activated = seeds

            while len(new_activated) > 0:
                activated = new_activated
                new_activated = []
                for active in activated:
                    for neighborg in active.get_neighborgs():
                        if live_edges[active.get_id()][neighborg.get_id()] and not neighborg.is_active():
                            nodes_activ_prob[neighborg.get_id()] += 1
                            neighborg.set_active()
                            new_activated.append(neighborg)

        nodes_activ_prob = nodes_activ_prob / max_repetition
        return nodes_activ_prob
