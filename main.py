import math

import matplotlib.pyplot as plt
import numpy as np
from network import Graph
from tqdm import tqdm
from mab import (Environment, Non_Stationary_Environment, SWTS_Learner,
                 TS_Learner)


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


# The function plots the approximation error: for each experiment we plot the spread given by the greedy algorithm
# K grows polynomially.
def approximation_error(graph, budget, scale_factor, num_experiments):
    plot_dict = {}
    plot_dict_2 = {}
    real_spread = greedy_algorithm(graph, budget, num_experiments)[1]
    k = 1
    for _ in range(0, num_experiments-1):
        print("Iteration: " + str(_ + 1) + "/" +
              str(num_experiments) + " | K = " + str(k), end="")

        seeds, spread = greedy_algorithm(graph, budget, k)

        plot_dict[k] = spread
        plot_dict_2[k] = real_spread
        k = math.ceil(k * scale_factor)
        print("", end="\r")

    print("", end="")
    plot_dict[k] = real_spread
    plot_dict_2[k] = real_spread

    lists = sorted(plot_dict.items())
    lists_2 = sorted(plot_dict_2.items())
    x, y = zip(*lists)
    x_2, y_2 = zip(*lists_2)

    plt.plot(x, y, label='Approximated Spread', color='tab:blue', linestyle='-')
    plt.plot(x_2, y_2, label='Real Spread', color='tab:orange', linestyle='--')
    plt.title("Graph " + str(graph.id) + ": Social Influence Maximization - Approximation Error")
    plt.ylabel("Activation Spread")
    plt.xlabel("Montecarlo Iterations")
    plt.legend()

    plt.show()


# The same function above but we have here multiple graphs and we call the cumulative algorithm
def cumulative_approximation_error(graphs, budget, scale_factor, num_experiments):
    plot_dict = {}
    plot_dict_2 = {}
    real_spread = sum(cumulative_greedy_algorithm(graphs, budget, 500)[1].values())
    k = 1
    for _ in range(0, num_experiments):
        print("Iteration: " + str(_ + 1) + "/" +
              str(num_experiments) + " | K = " + str(k), end="")

        seeds, spreads = cumulative_greedy_algorithm(graphs, budget, k)

        # y axis of the plot shows the cumulative_spread (sum of spread of each single graph)
        plot_dict[k] = sum(spreads.values())
        plot_dict_2[k] = real_spread
        k = math.ceil(k * scale_factor)
        print("", end="\r")

    print("", end="")

    lists = sorted(plot_dict.items())
    lists_2 = sorted(plot_dict_2.items())
    x, y = zip(*lists)
    x_2, y_2 = zip(*lists_2)

    plt.plot(x, y, label='Approximated Spread', color='tab:blue', linestyle='-')
    plt.plot(x_2, y_2, label='Spread after 500 repetitions', color='tab:orange', linestyle='--')
    plt.title("Cumulative Social Influence Maximization - Approximation Error")
    plt.ylabel("Activation Spread")
    plt.xlabel("Montecarlo Iterations")
    plt.legend()

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


# Function to avoid repetitions in the code
def get_beta_update_variables(indeces, i, graph):
    x = indeces[0][i]
    y = indeces[1][i]
    alpha = graph.beta_parameters_matrix[x][y].a
    beta = graph.beta_parameters_matrix[x][y].b

    mu = np.random.beta(alpha, beta)
    return x, y, alpha, beta, mu


# The function returns the seed set that we will use in point 4.
# We try to perform both exploitation of available info and exploration
# (with probability epsilon) of new possibilities (updating alfa and beta parameters)
def choose_seeds(graph, budget, epsilon, simulations):
    z = np.random.binomial(1, epsilon)
    if z == 0:
        # Exploit the available information
        seeds, _ = greedy_algorithm(graph, budget, simulations)
    else:
        # Find the position of the existing edges
        indeces = np.where(graph.adj_matrix > 0)

        # Retrieve for each of them alpha and beta, compute the deviation and update probability
        for i in range(len(indeces[0])):
            x, y, alpha, beta, mu = get_beta_update_variables(indeces, i, graph)
            sigma = (1 / (alpha + beta)) * np.sqrt((alpha * beta) / (alpha + beta + 1))
            graph.adj_matrix[x][y] = mu + sigma

        # print(graph.adj_matrix)
        seeds, _ = greedy_algorithm(graph, budget, simulations)
    return seeds

def choose_seeds_from_sampling(graph, budget, simulations):
    indeces = np.where(graph.adj_matrix > 0)

    # Retrieve for each of them alpha and beta, compute the deviation and update probability
    for i in range(len(indeces[0])):
        x, y, alpha, beta, mu = get_beta_update_variables(indeces, i, graph)
        graph.adj_matrix[x][y] = mu
    seeds, _ = greedy_algorithm(graph, budget, simulations)

    return seeds


# Given the 2 graphs we compute the absolute value of the difference between the probabilities.
# We return the mean.
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


def point4(true_graph: Graph, budget, repetitions, simulations):
    # Copy the original graph
    graph = Graph(copy=true_graph)
    graph.adj_matrix = np.where(true_graph.adj_matrix > 0, 0.5, 0)

    x_list = []
    x2_list = []
    y_list = []
    y2_list = []

    total_error = 0.0

    # Main procedure
    for r in range(repetitions):
        print("Iteration: " + str(r + 1) + "/" + str(repetitions), end="")
        #epsilon = (1 - r / repetitions) ** 2
        seeds = choose_seeds_from_sampling(graph, budget, simulations)
        graph.influence_episode(seeds, true_graph.adj_matrix)

        # in this case where return only of the indices of the non-zero value
        indices = np.where(graph.adj_matrix > 0)

        error = get_total_error(graph, true_graph)
        total_error += error

        x_list.append(r)
        x2_list.append(r)
        y_list.append(total_error)
        y2_list.append(0)
        print("", end="\r")
    print("", end="")

    plt.plot(x_list, y_list, label='Bandit Approximation', color='tab:blue', linestyle='-')
    plt.plot(x2_list, y2_list, label='Ideal 0 Value', color='tab:orange', linestyle='--')
    plt.title("Unknown Activation Probabilities - Approximation Error")
    plt.ylabel("Approximation Error")
    plt.xlabel("Time")
    plt.legend()

    plt.show()


# Generate randomly the conversion rates
def generate_conversion_rate(prices):
    val = np.random.uniform(size=(len(prices)))
    conversion_rates = np.sort(val)[::-1]
    return conversion_rates

def point5(graphs, prices, conv_rates, k, budget, n_experiments, T):
    # init revenue and n_customer for each graph, expeeriment and day
    revenue = np.zeros([len(graphs), n_experiments, T])
    n_customers = np.zeros([len(graphs), n_experiments, T])
    revenue_per_price = np.zeros([len(graphs), n_experiments, len(prices)])

    best_best_graphs_seeds = []

    for g in range(len(graphs)):
        seeds, _ = greedy_algorithm(graphs[g], budget, k)
        best_best_graphs_seeds.append(seeds)

    for exper in tqdm(range(n_experiments)):
        # print(f'experiment : {exper + 1}/{n_experiments}')
        for g in range(len(graphs)):
            print(f'graph : {g + 1}/{len(graphs)}')
            learner = TS_Learner(n_arms=len(prices), arms=prices)
            env = Environment(len(prices), probabilities=conv_rates[g])

            for t in range(T):
                r = 0      # actual revenue of day t
                potential_customers = graphs[g].social_influence(best_best_graphs_seeds[g])
                # every day the seller does social influence
                n_customers[g][exper][t] = potential_customers

                for _ in range(potential_customers):
                    pulled_arm = learner.pull_arm()
                    reward = env.round(pulled_arm)
                    learner.update(pulled_arm, reward)
                    r += prices[pulled_arm] * reward

                revenue[g, exper, t] = r

            # compute revenue of each arm da printare (facoltativo)
            for arm in range(len(prices)):
                purchases = np.sum((np.array(learner.pulled_arms) == arm) * (np.array(learner.rewards)))
                revenue_arm = purchases * prices[arm]
                revenue_per_price[g][exper][arm] = revenue_arm

    # average over experiments
    avg_revenue = np.average(revenue, 1)
    avg_customers = np.average(n_customers, 1)
    avg_revenue_per_price = np.average(revenue_per_price, 1)

    # print the revenue for each price and graph
    print(prices)
    for g in range(len(graphs)):
        print(g, ':', list(avg_revenue_per_price[g]))

    # compute the cumulative true expected revenue
    true_expect_revenue = np.zeros([len(graphs), len(prices)])
    for g, conv_rate in enumerate(conv_rates):

        true_expect_revenue[g] = conv_rate*prices

    time = range(T)
    for g in range(len(graphs)):
        opt_revenue = []
        actual_revenue = []
        regret = []
        for day in range(T):
            # compute the clairvoyant revenue
            avg_customers_per_graph = np.mean(avg_customers, 1)
            opt = np.max(true_expect_revenue[g]) * avg_customers_per_graph[g]
            # revenue of the algorithm
            actual = avg_revenue[g][day]
            # compute the instantaneous regret
            regret.append(opt - actual)
            opt_revenue.append(opt)
            actual_revenue.append(actual)
        # print the instantaneous revenue
        plt.figure(1)
        ax1 = plt.subplot(221)
        ax1.set_title(f'Graph {g}: Instantaneous Revenue')
        plt.plot(time, actual_revenue, label='TS_SW')
        plt.plot(time, opt_revenue, '--', label='clairvoyant')
        plt.ylabel('revenue')
        plt.xlabel('Time Horizon')
        plt.legend(loc="lower right")

        # print the cumulative revenue
        ax2 = plt.subplot(222)
        ax2.set_title(f'Graph {g}: Cumulative Revenue')
        plt.plot(time, np.cumsum(actual_revenue), label='TS_SW')
        plt.plot(time, np.cumsum(opt_revenue), '--', label='clairvoyant')
        plt.xlabel('Time Horizon')
        plt.ylabel('revenue')
        plt.legend(loc="lower right")

        # print the cumulative regret
        ax3 = plt.subplot(223)
        ax3.set_title(f'Graph {g}: Cumulative Regret')
        plt.plot(time, np.cumsum(regret), label='TS_SW')
        plt.legend(loc="lower right")
        plt.xlabel('Time Horizon')
        plt.ylabel('regret')
        # plt.savefig(f'results/point5 graph{g+1}')
        plt.show()



def point6(graphs, prices, conv_rates, n_phases, k, budget, n_experiments, T):
    window_size = 2 * int((np.sqrt(T)))
    # init revenue and n_customer for each graph, expeeriment and day
    revenue = np.zeros([len(graphs), n_experiments, T])
    n_customers = np.zeros([len(graphs), n_experiments, T])
    best_graphs_seeds = []

    for g in range(len(graphs)):
        seeds, _ = greedy_algorithm(graphs[g], budget, k)
        best_graphs_seeds.append(seeds)

    for exper in tqdm(range(n_experiments)):
        # print(f'experiment : {exper + 1}/{n_experiments}')
        for g in range(len(graphs)):
            print(f'graph : {g + 1}/{len(graphs)}')
            learner = SWTS_Learner(len(prices), prices, window_size, T)
            env = Non_Stationary_Environment(len(prices), conv_rates[g], T)

            for t in range(T):
                r = 0  # actual revenue of day t
                # every day the sellers make social influence

                potential_customers = graphs[g].social_influence(best_graphs_seeds[g])

                n_customers[g][exper][t] = potential_customers

                for _ in range(potential_customers):
                    pulled_arm = learner.pull_arm()
                    reward = env.round(pulled_arm, t)
                    learner.update(pulled_arm, reward, t)
                    r += prices[pulled_arm] * reward
                # revenue of the day
                revenue[g, exper, t] = r

    # average over experiments
    avg_revenue = np.average(revenue, 1)
    avg_customers = np.average(n_customers, 1)

    # compute the cumulative true expected revenue
    true_expect_revenue = np.zeros([len(graphs), n_phases, len(prices)])
    for g, conv_rate in enumerate(conv_rates):
        for phase in range(n_phases):
            true_expect_revenue[g][phase] = conv_rate[phase] * prices

    time = range(T)
    for g in range(len(graphs)):
        opt_revenue = []
        actual_revenue = []
        regret = []
        for day in range(T):
            phase_size = T / n_phases
            curr_phase = int(day / phase_size)
            # compute the clairvoyant revenue
            avg_customers_per_graph = np.mean(avg_customers, 1)
            opt = np.max(true_expect_revenue[g][curr_phase]) * avg_customers_per_graph[g]
            # revenue of the algorithm
            actual = avg_revenue[g][day]
            # compute the instantaneous regret
            regret.append(opt - actual)
            opt_revenue.append(opt)
            actual_revenue.append(actual)

        # print the instantaneous revenue
        plt.figure(1)
        ax1 = plt.subplot(221)
        ax1.set_title(f'Graph {g}: Instantaneous Revenue')
        plt.plot(time, actual_revenue, label='TS_SW')
        plt.plot(time, opt_revenue, '--', label='clairvoyant')
        plt.ylabel('revenue')
        plt.xlabel('Time Horizon')
        plt.legend(loc="lower right")

        # print the cumulative revenue
        ax2 = plt.subplot(222)
        ax2.set_title(f'Graph {g}: Cumulative Revenue')
        plt.plot(time, np.cumsum(actual_revenue), label='TS_SW')
        plt.plot(time, np.cumsum(opt_revenue), '--', label='clairvoyant')
        plt.xlabel('Time Horizon')
        plt.ylabel('revenue')
        plt.legend(loc="lower right")

        # print the cumulative regret
        ax3 = plt.subplot(223)
        ax3.set_title(f'Graph {g}: Cumulative Regret')
        plt.plot(time, np.cumsum(regret), label='TS_SW')
        plt.legend(loc="lower right")
        plt.xlabel('Time Horizon')
        plt.ylabel('regret')
        # plt.savefig(f'results/point5 graph{g+1}')
        plt.show()


def point7(graphs, prices, conv_rates, n_phases, k, budget, n_experiments, T, simulations):
    window_size = 2 * int((np.sqrt(T)))
    # init revenue and n_customer for each graph, expeeriment and day
    revenue = np.zeros([len(graphs), n_experiments, T])
    n_customers = np.zeros([len(graphs), n_experiments, T])

    phases_lens = np.zeros([len(graphs), n_phases], dtype=int)

    best_graphs_seeds = []
    for g in range(len(graphs)):
        seeds, _ = greedy_algorithm(graphs[g], budget, k)
        best_graphs_seeds.append(seeds)

    for exper in range(n_experiments):
        for g in range(len(graphs)):
            learner = SWTS_Learner(len(prices), prices, window_size, T)
            env = Non_Stationary_Environment(len(prices), conv_rates[g], T)
            # init the graph for point 4
            graph = Graph(copy=graphs[g])
            graph.adj_matrix = np.where(graphs[g].adj_matrix > 0, 0.5, 0)

            print(f'Experiment : {exper+1}/{n_experiments} Graph : {g+1}/{len(graphs)}')
            for t in tqdm(range(T)):
                r = 0
                # every day the sellers make social influence
                seeds = choose_seeds_from_sampling(graph, budget, simulations)
                potential_customers = graph.influence_episode(seeds, graphs[g].adj_matrix)
                best_potential_customers = graph.influence_episode(best_graphs_seeds[g], graphs[g].adj_matrix, sampling=False)
                indeces = np.where(graph.adj_matrix > 0)

                curr_phase = int(t / (T / n_phases))
                phases_lens[g][curr_phase] += potential_customers

                # Retrieve for each of them alpha and beta, compute the deviation and update probability
                for i in range(len(indeces[0])):
                    x = indeces[0][i]
                    y = indeces[1][i]
                    alpha = graph.beta_parameters_matrix[x][y].a
                    beta = graph.beta_parameters_matrix[x][y].b
                    mu = alpha / (alpha + beta)
                    graph.adj_matrix[x][y] = mu

                n_customers[g][exper][t] = best_potential_customers

                for _ in range(potential_customers):
                    pulled_arm = learner.pull_arm()
                    reward = env.round(pulled_arm, t)
                    learner.update(pulled_arm, reward, t)
                    r += prices[pulled_arm] * reward

                # revenue of the day
                revenue[g, exper, t] = r

    # average over experiments
    avg_revenue = np.average(revenue, 1)
    avg_customers = np.average(n_customers, 1)

    # compute the true expected revenue
    true_expect_revenue = np.zeros([len(graphs), n_phases, len(prices)])
    for g, conv_rate in enumerate(conv_rates):
        for phase in range(n_phases):
            true_expect_revenue[g][phase] = conv_rate[phase] * prices

    time = range(T)

    for g in range(len(graphs)):
        opt_revenue = []
        actual_revenue = []
        regret = []

        for day in range(T):
            phase_size = T / n_phases
            curr_phase = int(day / phase_size)
            # compute the clairvoyant revenue
            avg_customers_per_graph = np.mean(avg_customers, 1)
            opt = np.max(true_expect_revenue[g][curr_phase]) * avg_customers_per_graph[g]
            # revenue of the algorithm
            actual = avg_revenue[g][day]
            # compute the instantaneous regret
            regret.append(opt - actual)
            opt_revenue.append(opt)
            actual_revenue.append(actual)
        # print the instantaneous revenue
        plt.figure(1)
        ax1 = plt.subplot(221)
        ax1.set_title(f'Graph {g}: Instantaneous Revenue')
        plt.plot(time, actual_revenue, label='TS_SW')
        plt.plot(time, opt_revenue, '--', label='clairvoyant')
        plt.ylabel('revenue')
        plt.xlabel('Time Horizon')
        plt.legend(loc="lower right")

        # print the cumulative revenue
        ax2 = plt.subplot(222)
        ax2.set_title(f'Graph {g}: Cumulative Revenue')
        plt.plot(time, np.cumsum(actual_revenue), label='TS_SW')
        plt.plot(time, np.cumsum(opt_revenue), '--', label='clairvoyant')
        plt.xlabel('Time Horizon')
        plt.ylabel('revenue')
        plt.legend(loc="lower right")

        # print the cumulative regret
        ax3 = plt.subplot(223)
        ax3.set_title(f'Graph {g}: Cumulative Regret')
        plt.plot(time, np.cumsum(regret), label='TS_SW')
        plt.legend(loc="lower right")
        plt.xlabel('Time Horizon')
        plt.ylabel('regret')
        # plt.savefig(f'results/point5 graph{g+1}')
        plt.show()


points = [2, 3, 4, 5, 6, 7]

for point in points:
    if point is 2:
        graphs = [Graph(100, 0.2), Graph(150, 0.2), Graph(200, 0.1)]
        budget = 4
        scale_factor = 1.001
        num_experiments = 50

        point2(graphs, budget, scale_factor, num_experiments)

    # -----------------------------------------------------------------------------
    if point is 3:
        graphs = [Graph(100, 0.2), Graph(150, 0.2), Graph(200, 0.1)]
        budget = 4
        scale_factor = 1.001
        num_experiments = 50

        point3(graphs, budget, scale_factor, num_experiments)

    # -----------------------------------------------------------------------------
    if point is 4:
        graph = Graph(500, 0.005)
        budget = 5
        repetitions = 1000
        num_experiments = 10

        point4(graph, budget, repetitions, num_experiments)

    # -----------------------------------------------------------------------------
    if point is 5:
        # graphs = [Graph(300, 0.08), Graph(250, 0.08), Graph(350, 0.07)]
        graphs = [Graph(100, 0.05), Graph(125, 0.05), Graph(150, 0.05)]
        budget = 3
        k = 100  # number of montecarlo iterations
        n_experiments = 200
        time_horizon = 70
        prices = [500, 690, 750, 850]
        conv_rates = [generate_conversion_rate(prices) for g in graphs]   # each social network has its conv_rate

        point5(graphs, prices, conv_rates, k, budget, n_experiments, time_horizon)

    # -----------------------------------------------------------------------------
    if point is 6:
        # graphs = [Graph(300, 0.08), Graph(250, 0.08), Graph(350, 0.07)]
        graphs = [Graph(100, 0.05), Graph(125, 0.05), Graph(150, 0.05)]
        budget = 3
        k = 100  # number of montecarlo iterations
        n_experiments = 100
        time_horizon = 70
        n_phases = 3
        prices = [500, 690, 750, 850]
        # each social network has its conv_rate for each phase
        conv_rates = [[generate_conversion_rate(prices) for phase in range(n_phases)] for g in graphs]
        conv_rates = np.array(conv_rates)
        point6(graphs, prices, conv_rates, n_phases, k, budget, n_experiments, time_horizon)

    # -----------------------------------------------------------------------------
    if point is 7:
        # graphs = [Graph(50, 0.08), Graph(50, 0.08), Graph(30, 0.07)]
        graphs = [Graph(40, 0.3), Graph(40, 0.3), Graph(40, 0.3)]
        budget = 3
        k = 80
        simulations = 5
        n_phases = 3
        time_horizon = 70
        n_experiments = 20
        prices = [500, 690, 750, 850]
        # each social network has its conv_rate for each phase
        conv_rates = [[generate_conversion_rate(prices) for phase in range(n_phases)] for g in graphs]
        conv_rates = np.array(conv_rates)
        point7(graphs, prices, conv_rates, n_phases, k, budget, n_experiments, time_horizon, simulations)
