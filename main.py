import numpy as np
from matplotlib.pylab import plt
from network import Graph
from tqdm import tqdm


def greedy_algorithm(graph, budget, k, verbose=True):
    seeds = []
    spreads = []
    nodes = graph.nodes.copy()
    print("\n----------------------Starting greedy algorithm with:\nBudget: " + str(budget) + ", k: " + str(k))
    for _ in range(budget):
        if verbose:
            print("\n////////////////////////////////////////////////")
            print("\nInteraction cycle number " + str(_))

        best_spread = 0

        # For all the nodes which are not seed
        for node in tqdm(nodes):
            spread = graph.monte_carlo_sampling(seeds + [node], k, verbose)

            if spread > best_spread:
                best_spread = spread
                best_node = node

        spreads.append(best_spread)
        seeds.append(best_node)

        if verbose:
            print("-----------------------------------------------")
            print("The nodes are: ")
            for node in nodes:
                print(node)
            print(f"\nThe best node was: {best_node}, with spread equal to {best_spread}")
            print("-----------------------------------------------")

        # I remove it from nodes in order to not evaluate it again in the future
        if nodes:
            nodes.remove(best_node)

        if verbose:
            print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n")
    return seeds, spreads


my_graph = Graph(200, 0.05)
max_experiment = 10
plotDict = {}
k = 2
scale_factor = 0.72
for _ in range(0, max_experiment):
    k = k + int(k * scale_factor)
    seeds, spreads = greedy_algorithm(my_graph, 3, k, verbose=False)
    print(spreads)
    plotDict[k] = spreads[-1]


lists = sorted(plotDict.items())
x, y = zip(*lists)

plt.plot(x, y)
plt.show()

verbose = False
if verbose:
    print(spreads)
    print("")
    print("********FINAL SEED SET********")
    for seed in seeds:
        print(seed)

    for spread in spreads:
        print(spread)
