# IMPORTS:
import numpy as np


# ALPHA function:
# probability, given the prior seq. of symbols (x_0 -> x_{t+1}), that state i appears at position (t+1)
def alpha(i, t_plus_1, a_dist):
    if t_plus_1 == 0:
        prob = emissions[i][X[0]] * initials[i]

        return prob
    else:
        prob_sum = 0

        for j in range(0, N):
            prob_sum += a_dist[j][t_plus_1 - 1] * transitions[i][j]

        prob = emissions[i][X[t_plus_1]] * prob_sum

        return prob


# BETA function:
# probability of sequence (x_{t+1} -> x_{T - 1}), given preceding state i at position t
def beta(i, t, b_dist):
    if t == T - 1:
        prob = 1
        return prob
    else:
        prob = 0

        for j in range(0, N):
            prob += b_dist[j][t + 1] * transitions[i][j] * emissions[j][X[t + 1]]

        return prob


# GAMMA function:
# probability of state i appearing at position t
def gamma(i, t, A, B):
    numerator = A[i][t] * B[i][t]
    denominator = 0

    for j in range(0, N):
        denominator += A[j][t] * B[j][t]

    return numerator / denominator


# XI function:
# probability of state i appearing at position t and state j appearing at position (t + 1)
def xi(i, j, t, A, B):
    numerator = A[i][t] * transitions[i][j] * B[j][t + 1] * emissions[j][X[t + 1]]
    denominator = 0

    for p in range(0, N):
        for q in range(0, N):
            denominator += A[p][t] * transitions[p][q] * B[q][t + 1] * emissions[q][X[t + 1]]

    return numerator / denominator


# individual UPDATE function:
# update a single value in the arrays for initials, transitions, and emissions
def new_initial(i, alp, bet):
    initial_prob = gamma(i, 0, alp, bet)

    return initial_prob


def new_transition(i, j, alp, bet):
    numerator = 0
    denominator = 0

    for t in range(0, T - 1):
        numerator += xi(i, j, t, alp, bet)

    for t in range(0, T - 1):
        denominator += gamma(i, t, alp, bet)

    print(f"transition numerator = {numerator}")
    print(f"transition denominator = {denominator}\n")

    return numerator / denominator


def new_emission(i, x, alp, bet):
    numerator = 0
    denominator = 0

    for t in range(0, T):
        numerator += gamma(i, t, alp, bet) * int(X[t] == x)

    for t in range(0, T):
        denominator += gamma(i, t, alp, bet)

    return numerator / denominator


# INIT function:
# initialise parameter values - i.e. transitions, emissions, initials
def init():
    # initialise transition matrix M, emission matrix E, and initial matrix I as uniform distributions
    M = np.full((N, N), 1 / N)
    E = np.full((N, K), 1 / K)
    I = np.full(N, 1 / N)

    return M, E, I


# RUN function combining E-step and M-step
def run(observations, alphabet, no_states, its=1):
    # t refers to a position in the observation sequence X
    # T is the number of positions
    # K is the number of symbols in the alphabet of observations
    # N is the number of possible states

    # GLOBALS:
    global transitions
    global emissions
    global initials
    global X
    global T
    global K
    global N

    # initialise variables
    # X = observations
    # T = len(observations)
    # K = len(alphabet)
    # N = no_states
    # transitions, emissions, initials = init()

    X = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    T = len(X)
    alphabet = [0, 1]
    K = len(alphabet)
    N = 2

    transitions = [[0.5, 0.5],
                   [0.3, 0.7]]

    emissions = [[0.3, 0.7],
                 [0.8, 0.2]]

    initials = [0.2, 0.8]

    print("INITIALISATION!\n")
    print("Initials:\n", initials, end='\n\n')
    print("Transitions:\n", transitions, end='\n\n')
    print("Emissions:\n", emissions, end='\n\n')

    for it in range(its):
        # E-step: generate alpha/beta distributions
        alpha_dist = np.zeros((N, T))
        for position in range(T):
            for state in range(N):
                alpha_dist[state][position] = alpha(state, position, alpha_dist)

        beta_dist = np.zeros((N, T))
        for position in range(T - 1, -1, -1):
            for state in range(N):
                beta_dist[state][position] = beta(state, position, beta_dist)

        # print("Alpha:\n", alpha_dist, end='\n\n')
        # print("Beta:\n", beta_dist, end='\n\n')

        # M-step: update parameters using these distributions
        for state in range(N):
            initials[state] = new_initial(state, alpha_dist, beta_dist)

        for state1 in range(N):
            for state2 in range(N):
                transitions[state1][state2] = new_transition(state1, state2, alpha_dist, beta_dist)

        for state in range(N):
            for symbol in range(K):
                emissions[state][symbol] = new_emission(state, symbol, alpha_dist, beta_dist)

        print(f"ROUND {it}!\n")
        print("Initials:\n", initials, end='\n\n')
        print("Transitions:\n", transitions, end='\n\n')
        print("Emissions:\n", emissions, end='\n\n')


# example
run([0, 2, 0, 3, 1, 6, 2, 9, 2, 4, 8, 5, 1, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5)
