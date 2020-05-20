import numpy as np
from network_serna import Graph
from tqdm import tqdm

my_graph = Graph(20, 0.1)
seeds = my_graph.get_nodes()[1:2]
print(my_graph.monte_carlo_sampling(seeds, 10))


def greedy_algorithm(graph, budget):

    seeds = []
    # per spread intendo marginal
    spreads = []
    nodes = graph.get_nodes()
    for _ in tqdm(range(budget)):

        best_spread = 0
        for node in nodes:
            nodes_prob = graph.monte_carlo_sampling(seeds + [node], 10)
            # si può fare la media direttamente dentro monte carlo
            spread = np.mean(nodes_prob)

            if spread > best_spread:
                best_spread = spread
                best_node = node

        spreads.append(best_spread)
        seeds.append(best_node)

    return seeds, spreads


seeds, spreads = greedy_algorithm(my_graph, 20)

print(spreads)
