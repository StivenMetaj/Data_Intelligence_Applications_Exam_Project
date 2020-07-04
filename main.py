import numpy as np
from matplotlib.pylab import plt
from network import Graph
from tqdm import tqdm
import copy, math


def greedy_algorithm(graph, budget, k, verbose=False):
    seeds = []
    spreads = []
    nodes = graph.nodes.copy()
    #print("Budget: " + str(budget) + ", k: " + str(k))
    for _ in range(budget):
        if verbose:
            print("\n////////////////////////////////////////////////")
            print("\nInteraction cycle number " + str(_))

        best_spread = 0

        # For all the nodes which are not seed
        for node in nodes:
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
    seeds = {g: [] for g in graphs}
    spreads = {g: 0 for g in graphs}
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

                spread -= spreads[graph]  # compute marginal increase
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
        print('Best seeds:', [s.id for s in seeds],
              '\n')  # seeds that give the best influence executing k montecarlo iterations
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
        print('\nCumulative spread:',
              sum(spreads.values()))  # cumulative_spread of best configuration executing k montecarlo iterations
        for key, value in seeds.items():
            print(
                f'Graph: {key} \tseeds: {[el.id for el in value]}')  # seeds of each graph that give the best influence executing k montecarlo iterations
        plotDict[k] = sum(
            spreads.values())  # y axis of the plot shows the cumulative_spread (sum of spread of each single graph)

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
        print('Best seeds:',
              [s.id for s in seeds])  # seeds that give the best influence executing k montecarlo iterations

    print('\n------------------Approximation Error-------------------')
    # approximation error of the graphs (one at a time)
    for g in graphs:
        approximation_error(g, budget, num_experiment)


def point3(graphs, budget, k, num_experiment):
    print('\n---------------Cumulative Greedy algorithm---------------')

    # find the best cumulative seeds_set executing k montecarlo iterations
    seeds, spreads = cumulative_greedy_algorithm(graphs, budget, k, verbose=False)
    print('\nCumulative spread:',
          sum(spreads.values()))  # cumulative_spread of best configuration executing k montecarlo iterations
    print('\nBest seeds:')
    for key, value in seeds.items():
        print(
            f'\tGraph: {key.id} \tseeds: {[el.id for el in value]}')  # seeds of each graph that give the best influence executing k montecarlo iterations

    print('\n-------------Cumulative approximation error--------------')
    # cumulative approximation error of the graphs
    cumulative_approximation_error(graphs, budget, num_experiment)


def point4(true_graph, budget, repetitions, simulations):
    # Copy the original graph
    graph = copy.copy(true_graph)
    # But set the probabilities to uniform (0.5)
    graph.adj_matrix = np.where(true_graph.adj_matrix > 0, 0.5, 0)

    x_list = []
    y_list = []

    #print(true_graph.adj_matrix)
    records = []

    # Main procedure
    for r in tqdm(range(repetitions)):
        # Make epsilon decrease over time, many explorations at the beginning, many exploitations later
        epsilon = (1 - r / repetitions) ** 2
        seeds = choose_seeds(graph, budget, epsilon, simulations)
        graph.influence_episode(seeds, true_graph.adj_matrix)
        # test_seeds(graph, seeds)
        indeces = np.where(graph.adj_matrix > 0)

        # Retrieve for each of them alpha and beta, compute the deviation and update probability
        for i in range(len(indeces[0])):
            x = indeces[0][i]
            y = indeces[1][i]
            alpha = graph.beta_parameters_matrix[x][y].a
            beta = graph.beta_parameters_matrix[x][y].b
            mu = alpha / (alpha + beta)
            graph.adj_matrix[x][y] = mu

        error = get_total_error(graph, true_graph)

        x_list.append(r)
        y_list.append(error)
    print("-------TRUE MATRIX--------")
    print(true_graph.adj_matrix)
    print("-------ESTIMATED MATRIX--------")
    print(graph.adj_matrix)

    return  x_list, y_list


def choose_seeds(graph, budget, epsilon, simulations):
    z = np.random.binomial(1, epsilon)
    seeds = []
    if z == 0:
        # Exploit the available information
        seeds, _ = greedy_algorithm(graph, budget, simulations)
    else:
        # Find the position of the existing edges
        indeces = np.where(graph.adj_matrix > 0)

        # Retrieve for each of them alpha and beta, compute the deviation and update probability
        for i in range(len(indeces[0])):
            x = indeces[0][i]
            y = indeces[1][i]
            alpha = graph.beta_parameters_matrix[x][y].a
            beta = graph.beta_parameters_matrix[x][y].b
            mu = alpha / (alpha + beta)
            sigma = (1 / (alpha + beta)) * np.sqrt((alpha * beta) / (alpha + beta + 1))
            graph.adj_matrix[x][y] = mu + sigma

        # print(graph.adj_matrix)
        seeds, _ = greedy_algorithm(graph, budget, simulations)
    return seeds

def get_total_error(graph1: Graph, graph2: Graph):
    if len(graph1.nodes) == len(graph2.nodes):
        error = 0
        total_edges = 0
        for i in range(len(graph1.nodes)):
            for j in range(len(graph2.nodes)):
                if not math.isclose(graph1.adj_matrix[i][j], 0.0):
                    total_edges += 1
                    error += abs(graph1.adj_matrix[i][j] - graph2.adj_matrix[i][j])
        return error/total_edges


graph1 = Graph(100, 0.15)
# graph2 = Graph(250, 0.08)
# graph3 = Graph(350, 0.07)
# graphs = [graph1, graph2, graph3]

# budget = 3
# k = 10             # number of montecarlo iterations)
# num_experiment = 10

# point2(graphs, budget, k, num_experiment)
# point3(graphs, budget, k, num_experiment)

x, y = point4(graph1, 3, 1000, 10)
plt.plot(x, y)
plt.show()

