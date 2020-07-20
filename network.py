import random

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123456)


class Features:
    # Class containing all the features that each node will have
    def __init__(self, gender, age, location, interests):
        self.gender = gender
        self.age = age
        self.location = location
        self.interests = interests


class Node(object):
    # Class Node -> degree = neighbors.len()
    def __init__(self, id):
        self.id = id
        self.neighbors = []
        self.degree = 0
        self.features = self.create_features()
        self.beta_parameters = []

    def add_neighborg(self, node):
        self.neighbors.append(node)
        self.degree += 1

    def create_features(self):
        gender = (lambda x: "M" if x < 0.5 else "F")(np.random.uniform(0, 1))
        age = np.random.randint(5)
        location = np.array([1] * 1 + [0] * 5)
        np.random.shuffle(location)
        interests = np.random.randint(2, size=6).tolist()
        return Features(gender, age, location, interests)

    def __str__(self):
        return f"Node with ID: {self.id}, degree: {self.degree}"


class Beta(object):
    # Class containing the beta parameters for the learning algorithm (at the beginning uniform distribution)
    def __init__(self):
        self.a = 1
        self.b = 1
        self.played = 0

    def __str__(self):
        return f"Beta with alpha:{self.a}, beta:{self.b}, repetitions:{self.played}"


class Graph(object):
    id = 1  # graph id

    def __init__(self, n_nodes=1, connectivity=0.5, copy=None):
        if copy is None:
            self.constructor(n_nodes, connectivity)
        else:
            self.copy_constructor(copy)

    def constructor(self, n_nodes, connectivity):
        self.id = Graph.id
        Graph.id += 1

        self.n_nodes = n_nodes
        self.connectivity = connectivity
        self.adj_matrix = np.zeros([n_nodes, n_nodes], dtype=np.float)
        self.nodes = [Node(id) for id in range(self.n_nodes)]

        self.beta_parameters_matrix = np.empty((n_nodes, n_nodes), dtype=object)
        for i in np.ndindex(self.beta_parameters_matrix.shape): self.beta_parameters_matrix[i] = Beta()

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if np.random.rand() <= self.connectivity and i != j:
                    self.nodes[i].add_neighborg(self.nodes[j])
                    # Not uniform(0, 1) because we prefer low probabilities
                    features_value = self.evaluate_influence(self.nodes[i], self.nodes[j])
                    # features_value / how_much_low to normalize with respect to the uniform
                    how_much_low = 20
                    self.adj_matrix[i][j] = np.random.uniform(0, 1 / how_much_low) + (features_value / how_much_low)

    def copy_constructor(self, copy):
        self.id = copy.id
        self.n_nodes = copy.n_nodes
        self.connectivity = copy.connectivity
        self.adj_matrix = np.zeros([self.n_nodes, self.n_nodes], dtype=np.float)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.adj_matrix[i][j] = copy.adj_matrix[i][j]
        self.nodes = []
        for i in range(self.n_nodes):
            self.nodes.append(copy.nodes[i])
        self.beta_parameters_matrix = np.empty((self.n_nodes, self.n_nodes), dtype=object)
        for i in np.ndindex(self.beta_parameters_matrix.shape): self.beta_parameters_matrix[i] = Beta()

    # Evaluate influence of n1 over n2 (over the features)
    def evaluate_influence(self, n1, n2):
        authority = n1.degree / self.n_nodes

        gender_influence = \
            (lambda x, y: np.random.uniform(0.7, 1) if x == y else np.random.uniform(0, 0.7)) \
                (n1.features.gender, n2.features.gender)

        age_influence = \
            (lambda x, y: np.random.uniform(0, 1) / (abs(x - y) + 1)) \
                (n1.features.age, n2.features.age)

        location_influence = \
            (lambda x, y: np.random.uniform(0, 1) / (abs(np.where(x == 1)[0][0] - np.where(y == 1)[0][0]) + 1)) \
                (n1.features.location, n2.features.location)

        interests_influence = np.dot(n1.features.interests, n2.features.interests) / 6

        if self.id is 1:
            age_influence *= 2
        elif self.id is 2:
            interests_influence *= 2
        elif self.id is 3:
            authority *= 2

        total_influence = authority * (gender_influence + age_influence + interests_influence + location_influence)
        return total_influence

    # Given a set of seeds, the method returns the nodes that are active after the social influence
    def social_influence(self, seeds):
        active = []
        new_activated = seeds

        while new_activated:
            activated = new_activated
            new_activated = []
            for a in activated:
                neighbors = a.neighbors
                for n in neighbors:
                    if n not in active and np.random.binomial(1, self.adj_matrix[a.id][n.id]) == 1:
                        new_activated.append(n)
                        active.append(n)

        return len(active)

    def monte_carlo_sampling(self, seeds, max_repetition):
        nodes_activ_prob = np.zeros(self.n_nodes, dtype=np.float)

        for _ in range(max_repetition):

            # np.random.rand generates a matrix of random (0, 1) numbers!
            # We want a live_edges to work with
            live_edges = self.adj_matrix > np.random.rand(self.n_nodes, self.n_nodes)
            activated = []
            new_activated = seeds

            while new_activated:
                activated = new_activated + activated
                new_activated = []
                for active in activated:
                    for neighborg in active.neighbors:
                        if live_edges[active.id][neighborg.id] and not (
                                neighborg in new_activated or neighborg in activated):
                            nodes_activ_prob[neighborg.id] += 1
                            new_activated.append(neighborg)

        nodes_activ_prob = nodes_activ_prob / max_repetition

        return np.mean(nodes_activ_prob), np.std(nodes_activ_prob)

    def influence_episode(self, seeds, truth):
        binomial_matrix = np.zeros([self.n_nodes, self.n_nodes], dtype=np.float)
        indeces = np.where(self.adj_matrix > 0)

        for i in range(len(indeces[0])):
            x = indeces[0][i]
            y = indeces[1][i]
            true_probability = truth[x][y]
            r = np.random.binomial(1, true_probability)
            binomial_matrix[x][y] = r

        live_edges = binomial_matrix > 0

        activated = []
        new_activated = seeds

        while new_activated:
            activated = new_activated + activated
            new_activated = []
            for active in activated:
                for neighborg in active.neighbors:
                    if live_edges[active.id][neighborg.id] and not (
                            neighborg in new_activated or neighborg in activated):
                        self.beta_parameters_matrix[active.id][neighborg.id].a += 1
                        new_activated.append(neighborg)
                    else:
                        self.beta_parameters_matrix[active.id][neighborg.id].b += 1

        return len(activated) - len(seeds)

    def random_seeds(self, n=1):
        seeds = []
        for _ in range(n):
            seeds.append(random.choice(list(set(self.nodes) - set(seeds))))

        return seeds