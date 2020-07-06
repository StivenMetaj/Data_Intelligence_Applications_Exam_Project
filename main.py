import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from mab import Environment, Non_Stationary_Environment, TS_Learner
# perchè pylab?
from matplotlib.pylab import plt
from network import Graph
from scipy.stats import beta
from tqdm import tqdm


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


# graph1 = Graph(100, 0.15)
# graph2 = Graph(250, 0.08)
# graph3 = Graph(350, 0.07)
# graphs = [graph1, graph2, graph3]

# budget = 3
# k = 10             # number of montecarlo iterations)
# num_experiment = 10
# x, y = point4(graph1, 3, 1000, 10)
# plt.plot(x, y)
# plt.show()


def generate_conversion_rate(prices):
    val = np.random.uniform(size=(len(prices)))
    convertion_rates = np.sort(val)[::-1]

    return convertion_rates


def point5(graphs, prices, conv_rates):
    n_experiments = 50
    T = 50  # number of days
    # init revenue and n_customer for each graph, expeeriment and day
    revenue = np.zeros([len(graphs), n_experiments, T])
    n_customers = np.zeros([len(graphs), n_experiments, T])
    seller = [g.random_seeds(1) for g in graphs]
    revenue_per_price = np.zeros([len(graphs), n_experiments, len(prices)])

    for exper in range(n_experiments):
        for g in range(len(graphs)):
            learner = TS_Learner(n_arms=len(prices), arms=prices)
            env = Environment(len(prices), probabilities=conv_rates[g])

            for t in range(T):
                r = 0      # actual revenue of day t
                potential_customers = graphs[g].social_influence(seller[g])
                # every day the seller does social influence
                n_customers[g][exper][t] = len(potential_customers)

                for _ in potential_customers:
                    pulled_arm = learner.pull_arm()
                    reward = env.round(pulled_arm)
                    learner.update(pulled_arm, reward)
                    r += prices[pulled_arm] * reward
                revenue[g, exper, t] = r

            # compute revenue of each arm da printare (facoltativo)
            for arm in range(len(prices)):
                purchases = np.sum((np.array(learner.arm_pulled) == arm) * (np.array(learner.rewards)))
                revenue_arm = purchases * prices[arm]
                revenue_per_price[g][exper][arm] = revenue_arm

    # average over experiments
    avg_revenue = np.average(revenue, 1)
    avg_customers = np.average(n_customers, 1)
    avg_revenue_per_price = np.average(revenue_per_price, 1)

    # print the revenue for each price and graph
    print(prices)
    cumulative_revenue = np.sum(avg_revenue_per_price, 0)
    for g in range(len(graphs)):
        print(g, ':', list(avg_revenue_per_price[g]))

    # print the cumulative revenue for each price
    print(f'cumulative_revenue: {cumulative_revenue}')

    # compute the cumulative true expected revenue
    true_expect_revenue = np.zeros([len(graphs), len(prices)])
    for g, conv_rate in enumerate(conv_rates):
        true_expect_revenue[g] = conv_rate*prices
    cum_true_expected_revenue = np.sum(true_expect_revenue, 0)

    # print the best TS price and the true best price
    print(f'expected revenue per day: {cum_true_expected_revenue}')
    best_price = prices[np.argmax(cum_true_expected_revenue)]
    print(f'Best price: {best_price}')
    best_price_alg = prices[np.argmax(cumulative_revenue)]
    print(f'Best price algorithm: {best_price_alg}')

    time = range(T)
    cum_opt = np.zeros(T)
    cum_actual = np.zeros(T)
    cum_regret = np.zeros(T)
    for g in range(len(graphs)):
        opt_revenue = []
        actual_revenue = []
        regret = []
        for day in range(T):
            # compute the clairvoyant revenue
            opt = np.max(true_expect_revenue[g]) * avg_customers[g][day]
            # revenue of the algorithm
            actual = avg_revenue[g][day]
            # compute the instantaneous regret
            regret.append(opt - actual)
            opt_revenue.append(opt)
            actual_revenue.append(actual)
        # cumulative values over the graphs
        cum_regret += regret
        cum_actual += actual_revenue
        cum_opt += opt_revenue

    # print the cumulatives instantaneous rewards
    plt.plot(time, cum_opt)
    plt.plot(time, cum_actual)
    plt.show()

    # print the cumulative expected reward
    plt.plot(time, np.cumsum(cum_actual))
    plt.plot(time, np.cumsum(cum_opt))
    plt.show()

    # print the cumulative expected regret
    plt.plot(time, np.cumsum(cum_regret))
    plt.show()


graph1 = Graph(300, 0.08)
graph2 = Graph(250, 0.08)
graph3 = Graph(350, 0.07)
graphs = [graph1, graph2, graph3]

budget = 3
k = 10  # number of montecarlo iterations
num_experiment = 10

prices = [500, 690, 750, 850]
# conv_rates = {g: generate_conversion_rate(prices) for g in graphs} 
conv_rates = [generate_conversion_rate(prices) for g in graphs]   # each social network has its conv_rate

point5(graphs, prices, conv_rates)
