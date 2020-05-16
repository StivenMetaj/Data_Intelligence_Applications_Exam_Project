import random, json
import numpy as np

from queue import Queue
from itertools import permutations
from tqdm import tqdm

import networkx as nx
import matplotlib.pyplot as plt

# Considering these are our models
#
# Gender:
#        Male, Female
#
# Age:
#        <18, 18-25, 26-34, 35-50. >50
#
# Location:
#        Europe, Asia, North_America, Africa, South_America, Australia
#
# Interests:
#        Technology, Sports, Politics, Science, Economics, Health

random.seed(0)
distance_factors = {}
weights = [0.1, 0.15, 0.4, 0.25]
for i, feature in enumerate(["gender", "age", "location", "interests"]):
    distance_factors[feature] = weights[i]

noise = 0.05


class Features:
    def __init__(self, gender, age, location, interests):
        self.gender = gender
        self.age = age
        self.location = location
        self.interests = interests

    def get_dict(self):
        return {"gender": self.gender,
                "age": self.age,
                "location": self.location,
                "interests": self.interests
                }


class Node:
    def __init__(self, id):
        self.id = id
        self.activated = False

        self.features = self.create_features(id)

    def create_features(self, seed):
        np.random.seed(seed)

        gender = np.zeros(2)
        age = np.zeros(5)
        location = np.zeros(6)

        gender[np.random.randint(2)] = 1
        age[np.random.randint(5)] = 1
        location[np.random.randint(6)] = 1
        interests = np.random.randint(2, size=6)

        node_features = Features(gender.tolist(), age.tolist(), location.tolist(), interests.tolist())
        return node_features.get_dict()

    def __str__(self):
        temp = "ID: " + str(self.id) + "\nFeatures: "
        for key in self.features:
            temp = temp + "[" + str(key) + "] = " + str(self.features[key]) + " "
        return temp + "\n"


class Edge:
    def __init__(self, node1, node2):
        # It's not a good idea to save again the entire node as start or finish.
        self.start = node1
        self.finish = node2

        self.features_distance = self.calculate_features_distance(node1, node2)
        self.prob_of_activation = self.measure_similarity_distance()

        self.theta = np.random.dirichlet(np.ones(len(self.features_distance)), size=1).tolist()

    def calculate_features_distance(self, node1, node2):
        features_distance = {}
        features1 = node1.features
        features2 = node2.features

        # Map to convert into float type, instead of using int32 of numpy
        for feature in features1:
            features_distance[feature] = float(np.dot(features1[feature], features2[feature]))
        return features_distance

    def measure_similarity_distance(self):
        distance = 0
        for key, value in distance_factors.items():
            if key == "interests":
                distance += value * (1 - (self.features_distance[key] / 6))
            else:
                distance += value * (1 - self.features_distance[key])
        distance += noise
        return float(distance)

    def __str__(self):
        temp = "Node 1 ID: " + str(self.start.id) + \
               ", Node 2 ID: " + str(self.finish.id)
        temp = temp + "\nFeatures Distance: " + str(self.features_distance) + \
               ", Similarity Distance: " + str(self.prob_of_activation)
        return temp + "\n"


class Graph:
    def __init__(self, id, nodes_number, connectivity, budget):
        self.id = 100 * id
        self.nodes_number = nodes_number
        self.connectivity = connectivity
        self.nodes = []
        self.edges = []
        self.adjacent_matrix = [[0 for j in range(self.nodes_number)] for i in range(self.nodes_number)]

        self.budget = max(1, budget)

        self.create_nodes()
        self.connect_graph()

    def create_nodes(self):
        for i in range(self.nodes_number):
            self.nodes.append(Node(self.id + i))

    def connect_graph(self):
        for i in tqdm(range(self.nodes_number)):
            for j in range(i + 1, self.nodes_number):
                if random.random() <= self.connectivity:
                    self.edges.append(Edge(self.nodes[i], self.nodes[j]))
                    self.adjacent_matrix[i][j] = 1
                    self.adjacent_matrix[j][i] = 1

    def get_possible_seed_subsets(self):
        temp = []
        for number_of_seeds in tqdm(range(1, self.budget + 1)):
            seeds = list(permutations(self.nodes, number_of_seeds))
            for elem in seeds:
                temp.append(list(elem))
        return temp

    def get_live_edges(self):
        live_edges = []
        for edge in self.edges:
            if random.uniform(0, 1) < edge.prob_of_activation:
                live_edges.append(edge)
        return live_edges

    def get_activated_nodes(self, seeds, live_edges):
        activated_nodes = [x for x in seeds]
        changes = 1
        while changes is not 0:
            changes = 0
            for edge in live_edges:
                if edge.start in activated_nodes and edge.finish not in activated_nodes:
                    activated_nodes.append(edge.finish)
                    changes += 1
        return activated_nodes

    def monte_carlo_maximization(self, number_of_iterations):
        seeds = random.sample(self.nodes, self.budget)
        result = {}
        for node in self.nodes:
            result[node.id] = 0.0

        for i in tqdm(range(number_of_iterations)):
            live_edges = self.get_live_edges()
            activated_nodes = self.get_activated_nodes(seeds, live_edges)
            for node in activated_nodes:
                result[node.id] += 1.0

        for id in result:
            result[id] = float(float(result[id]) / number_of_iterations)
        return result

    def __str__(self):
        temp = "\n**********\nList of nodes:\n"
        for node in self.nodes:
            temp = temp + str(node)
        temp = temp + "List of edges:\n"
        for edge in self.edges:
            temp = temp + str(edge)
        return temp + "\n**********\n"

    def draw_graph(self):
        graph = nx.Graph()
        for node in self.nodes:
            graph.add_node(node.id)
            graph.nodes[node.id]['activated'] = node.activated

        for edge in self.edges:
            graph.add_edge(edge.start.id, edge.finish.id)
            graph.edges[edge.start.id, edge.finish.id]['prob'] = '%.3f' % (edge.prob_of_activation)

        pos = nx.spring_layout(graph)

        # nx.draw_networkx(graph, arrows=True, with_labels=True, node_size=500)
        # nx.spring_layout(graph)
        nx.draw(graph, pos)
        node_labels = nx.get_node_attributes(graph, 'activated')
        nx.draw_networkx_labels(graph, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(graph, 'prob')
        nx.draw_networkx_edge_labels(graph, pos, labels=edge_labels)

        plt.show()

    ## Function transforming the Graph class into a JSON serializable
    def turn_self_dict(self):
        for i in range(self.nodes_number):
            self.nodes[i] = self.nodes[i].__dict__
        self.nodes = {i: self.nodes[i] for i in range(len(self.nodes))}
        for i in range(len(self.edges)):
            # self.edges[i].start = self.edges[i].start.__dict__
            # self.edges[i].finish = self.edges[i].finish.__dict__
            self.edges[i] = self.edges[i].__dict__
        self.edges = {i: self.edges[i] for i in range(len(self.edges))}
        self.adjacent_matrix = {i: self.adjacent_matrix[i] for i in range(len(self.adjacent_matrix))}
        return self.__dict__

    ## Function used to generate a graph from JSON file
    def load_from_json(self, jsonfile):
        # recovering list of nodes from json
        for item in jsonfile.get('nodes').items():
            self.nodes.append(Node(item[1].get('id'), item[1].get('special_feature'), item[1].get('features')))
        # recovering adjacent matrix from json
        self.adjacent_matrix = []
        for item in jsonfile.get('adjacent_matrix').items():
            self.adjacent_matrix.append(item[1])
        # recovering list of edges from json
        for k, v in jsonfile.get('edges').items():
            node_head_id = v['start']
            node_tail_id = v['finish']
            self.edges.append(Edge(self.nodes[node_head_id], self.nodes[node_tail_id], v))