import numpy as np
from matplotlib.pylab import plt
from network import Graph
from tqdm import tqdm


def greedy_algorithm(graph, budget, k, verbose=False):
    seeds = []
    spreads = []
    nodes = graph.nodes.copy()
    print("Budget: " + str(budget) + ", k: " + str(k))
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
    return seeds, spreads[-1]

def cumulative_greedy_algorithm(graphs, budget, k, verbose=False):
    seeds = {g:[] for g in graphs}
    spreads = {g:0 for g in graphs}
    print("Budget: " + str(budget) + ", k: " + str(k))
    for _ in range(budget):
        if verbose:
            print("\n////////////////////////////////////////////////")
            print("\nInteraction cycle number " + str(_))

        best_spread = 0

        # For all the nodes which are not seed
        for graph in graphs:
            nodes = list(set(graph.nodes.copy()) - set(seeds[graph]))
            for node in tqdm(nodes):
                spread = graph.monte_carlo_sampling(seeds[graph] + [node], k, verbose)

                spread -= spreads[graph]          # compute marginal increase
                if spread > best_spread:
                    best_spread = spread
                    best_node = node
                    graph_best_node = graph

        spreads[graph_best_node] = best_spread
        seeds[graph_best_node].append(best_node)

        if verbose:
            print("-----------------------------------------------")
            print(f"\nThe best node was: {best_node} in graph: {graph_best_node}, with spread equal to {best_spread}")
            print("-----------------------------------------------")

        if verbose:
            print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n")

    return seeds, spreads

def approximation_error(graph, budget, num_experiment):
    plotDict = {}
    k = 2
    scale_factor = 0.72
    for _ in range(0, num_experiment):
        seeds, spread = greedy_algorithm(graph, budget, k, verbose=False)
        print('\nSpread:', spread)  # spread of best configuration executing k montecarlo iterations
        print('Best seeds:', [s.id for s in seeds], '\n')  # seeds that give the best influence executing k montecarlo iterations
        plotDict[k] = spread

        k = k + int(k * scale_factor)

    lists = sorted(plotDict.items())
    x, y = zip(*lists)

    plt.plot(x, y)
    plt.show()

def cumulative_approximation_error(graphs, budget, num_experiment):
    plotDict = {}
    k = 2
    scale_factor = 0.72
    for _ in range(0, num_experiment):
        seeds, spreads = cumulative_greedy_algorithm(graphs, budget, k, verbose=False)
        print('\nCumulative spread:', sum(spreads.values()))         # cumulative_spread of best configuration executing k montecarlo iterations
        for key, value in seeds.items():
            print(f'Graph: {key} \tseeds: {[el.id for el in value]}')     # seeds of each graph that give the best influence executing k montecarlo iterations
        plotDict[k] = sum(spreads.values())     # y axis of the plot shows the cumulative_spread (sum of spread of each single graph)

        k = k + int(k * scale_factor)

    lists = sorted(plotDict.items())
    x, y = zip(*lists)

    plt.plot(x, y)
    plt.show()

def point2(graphs, budget, k, num_experiment):
    print('\n--------------------Greedy algorithm---------------------')

    # find, for each graph, the best seeds_set executing k montecarlo iterations
    for g in graphs:
        print(f'\n------------Graph{g.id}------------')
        seeds, spread = greedy_algorithm(g, budget, k, verbose=False)
        print('\nSpread:', spread)  # spread of best configuration executing k montecarlo iterations
        print('Best seeds:', [s.id for s in seeds])  # seeds that give the best influence executing k montecarlo iterations

    print('\n------------------Approximation Error-------------------')
    # approximation error of the graphs (one at a time)
    for g in graphs:
        approximation_error(g, budget, num_experiment)

def point3(graphs, budget, k, num_experiment):
    print('\n---------------Cumulative Greedy algorithm---------------')

    # find the best cumulative seeds_set executing k montecarlo iterations
    seeds, spreads = cumulative_greedy_algorithm(graphs, budget, k, verbose=False)
    print('\nCumulative spread:', sum(spreads.values()))  # cumulative_spread of best configuration executing k montecarlo iterations
    print('\nBest seeds:')
    for key, value in seeds.items():
        print(f'\tGraph: {key.id} \tseeds: {[el.id for el in value]}')  # seeds of each graph that give the best influence executing k montecarlo iterations

    print('\n-------------Cumulative approximation error--------------')
    # cumulative approximation error of the graphs
    cumulative_approximation_error(graphs, budget, num_experiment)


graph1 = Graph(300, 0.1)
graph2 = Graph(250, 0.08)
graph3 = Graph(350, 0.07)
graphs = [graph1, graph2, graph3]

budget = 3
k = 10             # number of montecarlo iterations
num_experiment = 10

point2(graphs, budget, k, num_experiment)
point3(graphs, budget, k, num_experiment)
