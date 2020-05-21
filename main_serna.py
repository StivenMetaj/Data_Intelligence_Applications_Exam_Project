import numpy as np
from network_serna import Graph
from tqdm import tqdm

def greedy_algorithm(graph, budget):

    seeds = []
    spreads = []
    nodes = graph.nodes
    
    for _ in tqdm(range(budget)):
        print("Interaction cycle")

        best_spread = 0

        # For all the nodes which are not seed
        for node in nodes:
            spread = graph.monte_carlo_sampling(seeds + [node], 10)

            if spread > best_spread:
                best_spread = spread
                best_node = node

        spreads.append(best_spread)
        seeds.append(best_node)
        
        print("-----------------------------------------------")
        print("The nodes are: ")
        for node in nodes:
            print(node)
        print(f"The best node was: {best_node}")
        print("-----------------------------------------------")

        # I remove it from nodes in order to don't evaluate it again in the future
        if nodes:
            nodes.remove(best_node)
        
    return seeds, spreads

my_graph = Graph(14, 0.1)
seeds, spreads = greedy_algorithm(my_graph, 3)

print(spreads)
print("")
print("********FINAL SEED SET********")
for seed in seeds:
    print(seed)