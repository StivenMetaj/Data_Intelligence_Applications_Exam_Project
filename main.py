import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from mab import Environment, TS_Learner
# perchè pylab?
from matplotlib.pylab import plt
from network import Graph
from scipy.stats import beta
from tqdm import tqdm


# The function returns the best possible seeds set given a certain graph
def greedy_algorithm(graph, budget, k, verbose=False):
    seeds = []
    spreads = []
    best_node = None
    nodes = graph.nodes.copy()

    # In a cumulative way I compute the montecarlo sampling for each possibile new seed to see which one will be added
    for _ in range(budget):
        best_spread = 0

        # For all the nodes which are not seed
        for node in nodes:
            spread = graph.monte_carlo_sampling(seeds + [node], k)

            if spread > best_spread:
                best_spread = spread
                best_node = node

        spreads.append(best_spread)
        seeds.append(best_node)

        # I remove it from nodes in order to not evaluate it again in the future
        if nodes:
            nodes.remove(best_node)

    return seeds, spreads[-1]


# As before but we have multiple graphs and we decide the best seeds for each graph (they are correlated!)
def cumulative_greedy_algorithm(graphs, budget, k):
    seeds = {g: [] for g in graphs}
    spreads = {g: 0 for g in graphs}
    graph_best_node = None
    best_node = None

    for _ in range(budget):
        best_spread = 0

        # For all the nodes which are not seed
        for graph in graphs:
            # I want all the nodes that are not seeds!
            nodes = list(set(graph.nodes.copy()) - set(seeds[graph]))

            for node in nodes:
                spread = graph.monte_carlo_sampling(seeds[graph] + [node], k)

                spread -= spreads[graph]  # compute marginal increase
                if spread > best_spread:
                    best_spread = spread
                    best_node = node
                    graph_best_node = graph

        spreads[graph_best_node] = best_spread
        seeds[graph_best_node].append(best_node)

    return seeds, spreads


def approximation_error(graph, budget, scale_factor, num_experiments):
    plot_dict = {}
    k = 1
    for _ in range(0, num_experiments):
        print("Iteration: " + str(_ + 1) + "/" +
              str(num_experiments) + " | K = " + str(k), end="")

        seeds, spread = greedy_algorithm(graph, budget, k)

        plot_dict[k] = spread
        k = math.ceil(k * scale_factor)
        print("", end="\r")

    print("", end="")
    lists = sorted(plot_dict.items())
    x, y = zip(*lists)

    plt.plot(x, y)
    plt.show()


def cumulative_approximation_error(graphs, budget, scale_factor, num_experiments):
    plot_dict = {}
    k = 1
    for _ in range(0, num_experiments):
        print("Iteration: " + str(_ + 1) + "/" +
              str(num_experiments) + " | K = " + str(k), end="")

        seeds, spreads = cumulative_greedy_algorithm(graphs, budget, k)

        # y axis of the plot shows the cumulative_spread (sum of spread of each single graph)
        plot_dict[k] = sum(spreads.values())
        k = math.ceil(k * scale_factor)
        print("", end="\r")

    print("", end="")
    lists = sorted(plot_dict.items())
    x, y = zip(*lists)

    plt.plot(x, y)
    plt.show()


def point2(graphs, budget, scale_factor, num_experiments):
    print('\n--------------------Point 2---------------------')

    # approximation error of the graphs (one at a time)
    for _ in range(len(graphs)):
        print("--Graph " + str(_ + 1) + "--")
        approximation_error(graphs[_], budget, scale_factor, num_experiments)


def point3(graphs, budget, scale_factor, num_experiments):
    print('\n--------------------Point 3---------------------')

    # cumulative approximation error of the graphs
    cumulative_approximation_error(graphs, budget, scale_factor, num_experiments)


def point4(true_graph, budget, repetitions, simulations):
    # Copy the original graph
    graph = copy.copy(true_graph)

    # But set the probabilities to uniform (0.5)
    graph.adj_matrix = np.where(true_graph.adj_matrix > 0, 0.5, 0)

    x_list = []
    y_list = []

    # Main procedure
    for r in range(repetitions):
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

    return x_list, y_list


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
        return error / total_edges


def generate_conversion_rate(prices):
    val = np.random.uniform(size=(len(prices)))
    convertion_rates = np.sort(val)[::-1]

    return convertion_rates


def point5(graphs, prices, conv_rates):
    # print(f'convertion rates : {conv_rates}')
    n_experiments = 1
    daily_revenue = {g: {n: [] for n in range(n_experiments)} for g in graphs}
    daily_customers = {g: {n: [] for n in range(n_experiments)} for g in graphs}
    T = 50  # number of days
    graph_revenue = {g: {n: [] for n in range(n_experiments)} for g in graphs}
    seller = {g: g.random_seeds(1) for g in graphs}
    for n in range(n_experiments):
        for g in graphs:
            learner = TS_Learner(n_arms=len(prices), arms=prices)
            env = Environment(len(prices), probabilities=conv_rates[g])
            for t in range(T):
                r = 0  # actual revenue of day t
                potential_customers = g.social_influence(seller[g])  # every day the seller do social influence
                daily_customers[g][n].append(len(potential_customers))
                for _ in potential_customers:
                    pulled_arm = learner.pull_arm()
                    reward = env.round(pulled_arm)
                    learner.update(pulled_arm, reward)
                    r += prices[pulled_arm] * reward
                daily_revenue[g][n].append(r)
                '''if t == T-1:             # print beta distribution of the each arm
                    fig, ax = plt.subplots(nrows=2, ncols=2)
                    x = np.linspace(0, 1, 100)
                    ax[0, 0].plot(x, beta.pdf(x, learner.beta_param[0, 0], learner.beta_param[0, 1]))
                    ax[0, 1].plot(x, beta.pdf(x, learner.beta_param[1, 0], learner.beta_param[1, 1]))
                    ax[1, 0].plot(x, beta.pdf(x, learner.beta_param[2, 0], learner.beta_param[2, 1]))
                    ax[1, 1].plot(x, beta.pdf(x, learner.beta_param[3, 0], learner.beta_param[3, 1]))

                    plt.show()'''
            # print(learner.arm_pulled)
            # print(learner.rewards)
            revenue = 0
            for i in range(len(prices)):
                purchases = np.sum((np.array(learner.arm_pulled) == i) * (np.array(learner.rewards)))
                revenue = purchases * prices[i]
                # print(f'\trevenue :{revenue}')
                graph_revenue[g][n].append(revenue)

    avg_daily_revenue = {g: [] for g in graphs}
    avg_daily_customers = {g: [] for g in graphs}
    avg_graph_revenue = {g: [] for g in graphs}

    for g in graphs:
        for d in range(T):
            dr = 0
            dc = 0
            for n in range(n_experiments):
                dr += daily_revenue[g][n][d]
                dc += daily_customers[g][n][d]
            avg_daily_revenue[g].append(dr / n_experiments)
            avg_daily_customers[g].append(dc / n_experiments)
        for i in range(len(prices)):
            gr = 0
            for n in range(n_experiments):
                gr += graph_revenue[g][n][i]
            avg_graph_revenue[g].append(gr / n_experiments)

    daily_revenue = avg_daily_revenue
    daily_customers = avg_daily_customers
    graph_revenue = avg_graph_revenue

    print(prices)
    cumulative_revenue = [0] * len(prices)
    for i in range(len(prices)):
        for g in graphs:
            if i == 0:
                print(g.id, ':', graph_revenue[g])
            cumulative_revenue[i] += graph_revenue[g][i]

    print(f'cumulative_revenues: {cumulative_revenue}')
    rev = sorted(enumerate(cumulative_revenue), key=lambda x: x[1])[::-1]
    # print(rev)
    id_best_price = rev[0][0]
    best_price = prices[id_best_price]
    print(f'Best price: {best_price}')

    time = range(T)
    for g in graphs:
        opt_revenue = []
        actual_revenue = []
        regret = []
        for day in range(T):
            opt = best_price * daily_customers[g][day]  # optimal revenue of a specific day
            actual = daily_revenue[g][day]  # actual revenue of a specific day
            regret.append(opt - actual)  # regret
            opt_revenue.append(opt)
            actual_revenue.append(actual)
        plt.plot(time, opt_revenue)
        plt.plot(time, actual_revenue)
        plt.show()

        plt.plot(time, regret)
        plt.show()


# graph1 = Graph(300, 0.08)
# graph2 = Graph(250, 0.08)
# graph3 = Graph(350, 0.07)
# graphs = [graph1, graph2, graph3]
#
# budget = 3
# k = 10  # number of montecarlo iterations
# num_experiment = 10
#
# prices = [500, 600, 650, 700]
# conv_rates = {g: generate_conversion_rate(prices) for g in graphs}  # each social network has its conv_rate
#
# point5(graphs, prices, conv_rates)

# ------------------------------------------------------------------------------------------------------------------
graphs = [Graph(100, 0.2), Graph(125, 0.2), Graph(150, 0.2)]
budget = 3
scale_factor = 1.2
num_experiments = 15

#point3(graphs, budget, scale_factor, num_experiments)
x, y = point4(Graph(20, 0.1), budget, 100, 1)
plt.plot(x, y)
plt.show()