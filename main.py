import random

import matplotlib.pyplot as plt
import numpy as np
from mab import Environment, TS_Learner
from network import *
from scipy.stats import beta
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

def generate_conversion_rate(prices):
    val = np.random.uniform(size=(len(prices)))
    convertion_rates = np.sort(val)[::-1]

    return convertion_rates


def point5(graphs, prices, conv_rates):
    #print(f'convertion rates : {conv_rates}')
    n_experiments = 1
    daily_revenue = {g:{n:[] for n in range(n_experiments)} for g in graphs}
    daily_customers = {g:{n:[] for n in range(n_experiments)} for g in graphs}
    T = 50    # number of days
    graph_revenue = {g:{n:[] for n in range(n_experiments)} for g in graphs}
    seller = {g: g.random_seeds(1) for g in graphs}
    for n in range(n_experiments):
        for g in graphs:
            learner = TS_Learner(n_arms=len(prices), arms=prices)
            env = Environment(len(prices), probabilities=conv_rates[g])
            for t in range(T):
                r = 0      # actual revenue of day t
                potential_customers = g.social_influence(seller[g])        # every day the seller do social influence
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
            #print(learner.arm_pulled)
            #print(learner.rewards)
            revenue = 0
            for i in range(len(prices)):
                purchases = np.sum((np.array(learner.arm_pulled) == i) * (np.array(learner.rewards)))
                revenue = purchases * prices[i]
                #print(f'\trevenue :{revenue}')
                graph_revenue[g][n].append(revenue)

    avg_daily_revenue = {g:[] for g in graphs}
    avg_daily_customers = {g:[] for g in graphs}
    avg_graph_revenue = {g:[] for g in graphs}

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
            avg_graph_revenue[g].append(gr/n_experiments)

    daily_revenue = avg_daily_revenue
    daily_customers = avg_daily_customers
    graph_revenue = avg_graph_revenue

    print(prices)
    cumulative_revenue = [0]*len(prices)
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
            opt = best_price*daily_customers[g][day]      # optimal revenue of a specific day
            actual = daily_revenue[g][day]             # actual revenue of a specific day
            regret.append(opt - actual)              # regret
            opt_revenue.append(opt)
            actual_revenue.append(actual)
        plt.plot(time, opt_revenue)
        plt.plot(time, actual_revenue)
        plt.show()

        plt.plot(time, regret)
        plt.show()


graph1 = Graph(300, 0.08)
graph2 = Graph(250, 0.08)
graph3 = Graph(350, 0.07)
graphs = [graph1, graph2, graph3]

budget = 3
k = 10             # number of montecarlo iterations
num_experiment = 10

# point2(graphs, budget, k, num_experiment)
# point3(graphs, budget, k, num_experiment)

prices = [500, 600, 650, 700]
conv_rates = {g: generate_conversion_rate(prices) for g in graphs}    # each social network has its conv_rate

point5(graphs, prices, conv_rates)
