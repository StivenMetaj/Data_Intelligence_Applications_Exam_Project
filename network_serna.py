import numpy as np

np.random.seed(1234)



class Node(object):
    def __init__(self, id):
        self.id = id
        self.activated = False
        self.neighbors = []
        self.degree = 0

    def add_neighborg(self, node):
        self.neighbors.append(node)
        self.degree += 1

    def __str__(self):
        return f"Node with ID: {self.id}"

class Graph(object):
    def __init__(self, n_nodes, connectivity):

        self.n_nodes = n_nodes
        self.connectivity = connectivity
        self.adj_matrix = np.zeros([n_nodes, n_nodes], dtype=np.float)
        self.nodes = [Node(id) for id in range(self.n_nodes)]

        # Doubt in avoiding self connections
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if np.random.rand() <= self.connectivity and i != j:
                    self.nodes[i].add_neighborg(self.nodes[j])
                    self.adj_matrix[i][j] = np.random.rand()

    def monte_carlo_sampling(self, seeds, max_repetition):

        print("######################################")
        print("I'm receiving the following seeds for this MC sampling")
        for seed in seeds:
            print(seed)

        nodes_activ_prob = np.zeros(self.n_nodes, dtype=np.float)
        for _ in range(max_repetition):
            for node in self.nodes:
                if node in seeds:
                    node.activated = True
                else:
                    node.activated = False

            live_edges = self.adj_matrix > np.random.rand(self.n_nodes, self.n_nodes)
            new_activated = seeds

            while new_activated:
                activated = new_activated
                new_activated = []
                for active in activated:
                    for neighborg in active.neighbors:
                        if live_edges[active.id][neighborg.id] and not neighborg.activated:
                            nodes_activ_prob[neighborg.id] += 1
                            neighborg.activated = True
                            new_activated.append(neighborg)

        nodes_activ_prob = nodes_activ_prob / max_repetition

        print(f"Marginal increase of this soltution: {np.mean(nodes_activ_prob)}")
        return np.mean(nodes_activ_prob)
