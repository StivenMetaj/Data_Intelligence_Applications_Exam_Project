import numpy as np

#np.random.seed(12345)

class Features:
    def __init__(self, gender, age, location, interests):
        self.gender = gender
        self.age = age
        self.location = location
        self.interests = interests

class Node(object):
    def __init__(self, id):
        self.id = id
        self.activated = False
        self.neighbors = []
        self.degree = 0
        self.features = self.create_features()

    def add_neighborg(self, node):
        self.neighbors.append(node)
        self.degree += 1
    
    def create_features(self):
        gender = (lambda x: "M" if x < 0.5 else "F")(np.random.uniform(0,1))
        age = np.random.randint(5)
        location = np.array([1]*1 + [0]*5)
        np.random.shuffle(location)
        interests = np.random.randint(2, size=6).tolist()
        return Features(gender, age, location, interests)

    def __str__(self):
        return f"Node with ID: {self.id}, degree: {self.degree}"


class Graph(object):
    def __init__(self, n_nodes, connectivity):
        self.n_nodes = n_nodes
        self.connectivity = connectivity
        self.adj_matrix = np.zeros([n_nodes, n_nodes], dtype=np.float)
        self.nodes = [Node(id) for id in range(self.n_nodes)]

        # Doubt in avoiding self connections
        # Stiv: avoiding self connections mi gusta, ma dagli esempi sulle slide se i->j anche j->i... al massimo
        #       con probabilità diverse no?
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if np.random.rand() <= self.connectivity and i != j:
                    self.nodes[i].add_neighborg(self.nodes[j])
                    self.adj_matrix[i][j] = np.random.uniform(0, 0.1)

    # Evaluate influence of n1 over n2
    def evaluate_influence(self, n1, n2):
        authority = n1.degree/self.n_nodes
        gender_influence = (lambda x,y: np.random.uniform(0.7,1) if x == y else np.random.uniform(0,0.7)) \
                           (n1.features.gender,n2.features.gender)
        age_influence = (lambda x, y: np.random.uniform(0,1)/(abs(x-y)+1)) \
                        (n1.features.age, n2.features.age)
        location_influence = (lambda x,y: np.random.uniform(0,1)/(abs(np.where(x == 1)[0][0]-np.where(y == 1)[0][0])+1)) \
                             (n1.features.location, n2.features.location)
        interests_influence = np.dot(n1.features.interests, n2.features.interests)/6
        total_influence = authority*(gender_influence + age_influence + interests_influence)
        return total_influence
    
    def monte_carlo_sampling(self, seeds, max_repetition, verbose):
        if verbose:
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

        if verbose:
            print(f"Marginal increase of this soltution: {np.mean(nodes_activ_prob)}")
        return np.mean(nodes_activ_prob)