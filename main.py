import SocialNetwork as sn
import random
import numpy as np

networks = {}
seed = 0
random.seed(seed)

montecarlo_results = []

for id in range(1, 4):
    nodes_number = random.randint(10, 30)
    connectivity = random.uniform(0, 1)
    budget = random.randint(1, 5)

    networks[id] = sn.Graph(id, nodes_number, connectivity, budget)

    networks[id].draw_graph()

    epsilon = 0.05
    delta = 0.25

    iterations = int((1 / (epsilon ** 2)) * np.log(networks[id].budget + 1) * np.log(1 / delta))
    montecarlo_results.append(networks[id].monte_carlo_maximization(iterations))

print("ciao")


