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
        self.estimate_parameters = []

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
    id = 1       # graph id

    def __init__(self, n_nodes, connectivity):
        self.n_nodes = n_nodes
        self.connectivity = connectivity
        self.adj_matrix = np.zeros([n_nodes, n_nodes], dtype=np.float)
        self.nodes = [Node(id) for id in range(self.n_nodes)]
        self.id = Graph.id
        Graph.id += 1

        # Doubt in avoiding self connections
        # Stiv: avoiding self connections mi gusta, ma dagli esempi sulle slide se i->j anche j->i... al massimo
        #       con probabilità diverse no?
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if np.random.rand() <= self.connectivity and i != j:
                    self.nodes[i].add_neighborg(self.nodes[j])
                    self.adj_matrix[i][j] = np.random.uniform(0, 0.1)

    def init_estimates(self):
        for i in self.nodes:
            for j in range(i.degree):
                i.estimate_parameters = [[1, 1] for _ in range(i.degree)]

    def initializeUniformWeights(self):
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adj_matrix[i][j] != 0:
                    self.adj_matrix[i][j] = 1 / len(self.nodes[i].neighbors)


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

    # In output we have a list where each element has 2 sub-elements:
    # the first is the id of the activated node (the first k elements of the list will be the initial seeds!),
    # the second is a list of -1, 0, 1 , having length equal to the degree of the node ->
    #  0: the neighbor was unactive but the binomial gave us 0
    #  1: the neighbor was unactive and the binomial gave us 1
    # -1: the neighbor was already active!
    # WE WILL USE THIS LIST TO UPDATE THE BETA PARAMETERS FOR EACH EDGE
    def simulate_episode(self, seeds):
        realizations_per_node = []

        # I want the ids of the seeds
        seeds = [seed.id for seed in seeds]

        for i in seeds:
            self.nodes[i].activated = True

        for i in seeds:
            realizations = []
            for j in range(len(self.nodes)):
                if self.adj_matrix[i][j] != 0:
                    if not self.nodes[j].activated:
                        realization = np.random.binomial(1, self.adj_matrix[i][j])
                        if realization == 1:
                            seeds.append(j)
                            self.nodes[j].activated = True
                        realizations.append(realization)
                    else:
                        realizations.append(-1)

            realizations_per_node.append([i, realizations])

        nodes_activated = len(seeds)

        for id in seeds:
            self.nodes[id].activated = False

        return realizations_per_node, nodes_activated

    # Given the id of the node and its realizations of binomial random variables
    # we update the beta parameters of each edge of that node
    def update_estimations(self, id, realizations):
        temp = self.nodes[id].estimate_parameters
        for i in range(len(realizations)):
            if realizations[i] != -1:
                temp[i][0] += realizations[i]
                temp[i][1] += 1 - realizations[i]

    # We change the probabilities of our adj matrix following the estimated beta parameters!
    def update_weights(self):
        for i in self.nodes:
            id1 = i.id
            neighbor_id = 0
            for j in self.nodes:
                id2 = j.id
                if self.adj_matrix[id1][id2] != 0:
                    self.adj_matrix[id1][id2] = np.random.beta(a=i.estimate_parameters[neighbor_id][0],
                                                               b=i.estimate_parameters[neighbor_id][1])
                    neighbor_id += 1
