# IMPORTS:
import numpy as np


# ALPHA function:
# probability, given the prior seq. of symbols (x_0 -> x_{t+1}), that state i appears at position (t+1)
def alpha(i, t_plus_1, a_dist):
    if t_plus_1 == 0:
        return initials[i] * emissions[i][X[0]]
    else:
        prob_sum = 0

        for j in range(N):
            prob_sum += a_dist[j][t_plus_1 - 1] * emissions[i][X[t_plus_1]] * transitions[j][i]

        return prob_sum


# BETA function:
# probability of sequence (x_{t+1} -> x_{T - 1}), given preceding state i at position t
def beta(i, t, b_dist):
    if t == -1:
        return 1
    else:
        prob = 0

        for j in range(N):
            prob += b_dist[j][t + 1] * emissions[j][X[t + 1]] * transitions[i][j]

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

    res = numerator / denominator
    return res


# UPDATE functions:
# update values of initials, transitions, and emissions arrays
def new_initials(alp, bet):
    gamma_dist = np.zeros(N)
    for state in range(N):
        gamma_dist[state] = gamma(state, 0, alp, bet)

    return gamma_dist


def new_transitions(alp, bet):
    xi_dist = np.zeros((N, N, T-1))
    gamma_dist = np.zeros((N, T-1))
    for state1 in range(N):

        for t in range(0, T - 1):
            gamma_dist[state1][t] = gamma(state1, t, alp, bet)

        for state2 in range(N):
            for t in range(0, T - 1):
                xi_ijt = xi(state1, state2, t, alp, bet)
                xi_dist[state1][state2][t] = xi_ijt

    a = np.sum(xi_dist, 2) / np.sum(gamma_dist, axis=1).reshape((-1, 1))
    return a


def new_emissions(alp, bet):
    numerators = np.ones((N, K, T))
    denominators = np.ones((N, K, T))

    for state in range(N):
        for symbol in range(K):

            for t in range(0, T):
                gamma_t = gamma(state, t, alp, bet)
                numerators[state][symbol][t] = gamma_t * int(X[t] == symbol)
                denominators[state][symbol][t] = gamma_t

    a = np.sum(numerators, 2) / np.sum(denominators, 2)
    return a


def forwards():
    a = np.zeros((N, T))
    for position in range(T):
        for state in range(N):
            a[state][position] = alpha(state, position, a)

    return a


def backwards():
    b = np.zeros((N, T))
    for position in range(1, T + 1):
        for state in range(N):
            b[state][-position] = beta(state, -position, b)

    return b


# INIT function:
# initialise parameter values - i.e. transitions, emissions, initials
def init(no_states, obs, symbols):
    # number of observation positions T
    positions = len(obs)

    # size of alphabet K
    no_symbs = len(symbols)

    # transition matrix M
    M = np.full((no_states, no_states), 1/no_states)

    # emission matrix E
    E = []
    for state in range(1, no_states + 1):
        em = np.array([symb for symb in range(state, no_states * no_symbs + 1, no_states)])
        E.append(np.divide(em, em.sum()))
    E = np.array(E)

    # initial matrix I
    I = np.full(no_states, 1/no_states)

    return positions, no_symbs, M, E, I


def convergence(nt, ne, ni, epsilon=0.0001):
    del_t = np.max(np.abs(transitions - nt))
    del_e = np.max(np.abs(emissions - ne))
    del_i = np.max(np.abs(initials - ni))

    return (del_t < epsilon) and (del_e < epsilon) and (del_i < epsilon)


# RUN function combining E-step and M-step
def run(observations, alphabet, no_states, its=1):

    # GLOBALS:
    global N  # N is the number of possible states
    global X  # X is the observation sequence
    global T  # T is the number of positions in X
    global K  # K is the number of symbols in the alphabet
    # t refers to a position in the observation sequence X

    global transitions
    global emissions
    global initials

    # initialise variables
    N = no_states
    X = observations
    T, K, transitions, emissions, initials = init(N, X, alphabet)
    converged = False

    print(f"init_E: \n{emissions}")
    print(f"init_T: \n{transitions}")
    print(f"init_pi: \n{initials}")

    for it in range(its):
        # E-step: generate alpha/beta distributions
        alpha_dist = forwards()
        beta_dist = backwards()

        # print("Alpha:\n", alpha_dist, end='\n\n')
        # print("Beta:\n", beta_dist, end='\n\n')

        # M-step: update parameters using these distributions
        new_init = new_initials(alpha_dist, beta_dist)
        new_trans = new_transitions(alpha_dist, beta_dist)
        new_emiss = new_emissions(alpha_dist, beta_dist)

        converged = convergence(new_trans, new_emiss, new_init)

        initials = new_init
        transitions = new_trans
        emissions = new_emiss

        if converged:
            print(f"Converged at iteration: {it}")
            break

    return transitions, emissions, initials, converged


# example
seq = [0, 1, 2, 2, 0, 1, 2, 0]
symbs = [0, 1, 2]
states = 2
t, e, i, c = run(seq, symbs, states, its=2)

print(f"Convergence: {c}\n")
print("Initials:\n", i, end='\n\n')
print("Transitions:\n", t, end='\n\n')
print("Emissions:\n", e, end='\n\n')
