# IMPORTS:
import numpy as np

"""
DEPENDENCIES: numpy

INPUT:

To run EM algorithm (Baum-Welch) on an input sequence of observations call:
    
    run(observations, k, alphabet)

where `k` is the number of states (of type integer)
    e.g. k = 3

and `observations` is the sequence of observed letters

the sequence `observations` may be of type:
    a string of characters, where each character is an observed letter
         e.g. observations = 'pbafbza'
    OR a list/tuple, where each element of the list/tuple is an observed letter
        e.g. observations = ['p', 'b', 'a', 'f', 'b', 'z', 'a']
        
and `alphabet` is the collection of all possible observations 
(has to be a type with an associated length, where each element is a observation, e.g. string, list, or tuple)

OUTPUT:

The algorithm produces three outputs:

    transitions, emissions, initials
    
`transitions` is the `k` by `k` transition matrix:

        state0 state1 state2 ...
state0
state1
state2
...

`emissions` is the emission matrix:

        letter0 letter1 letter2 ...
state0
state1
state2
...

`initials` is the initial matrix:

        state0 state1 state2 ...
prob.


N.B. all other functions are for internal computations
"""


# ALPHA function:
# Probability, given the prior seq. of symbols (x_0 -> x_{t+1}), that state `i` appears at position `t+1`
def alpha(i, t_plus_1, a_dist):
    if t_plus_1 == 0:
        return initials[i] * emissions[i][X[0]]
    else:
        prob_sum = 0

        for j in range(N):
            prob_sum += a_dist[j][t_plus_1 - 1] * transitions[j][i]

        return prob_sum * emissions[i][X[t_plus_1]]


# BETA function:
# Probability of sequence (x_{t+1} -> x_{T - 1}), given preceding state `i` at position `t`
def beta(i, t, b_dist):
    if t == -1:
        return 1
    else:
        prob = 0

        for j in range(N):
            prob += b_dist[j][t + 1] * emissions[j][X[t + 1]] * transitions[i][j]

        return prob


# GAMMA function:
# Probability of state `i` appearing at position `t`
def gamma(i, t, A, B):
    numerator = A[i][t] * B[i][t]
    denominator = 0

    for j in range(0, N):
        denominator += A[j][t] * B[j][t]

    return numerator / denominator


# XI function:
# Probability of state `i` appearing at position `t` and state `j` appearing at position `t + 1`
def xi(i, j, t, A, B):
    numerator = A[i][t] * transitions[i][j] * B[j][t + 1] * emissions[j][X[t + 1]]
    denominator = 0

    for p in range(0, N):
        for q in range(0, N):
            denominator += A[p][t] * transitions[p][q] * B[q][t + 1] * emissions[q][X[t + 1]]

    res = numerator / denominator
    return res


# UPDATE functions:
# Calculate the new initial matrix
def new_initials(alp, bet):
    gamma_dist = np.zeros(N)
    for state in range(N):
        gamma_dist[state] = gamma(state, 0, alp, bet)

    return gamma_dist


# Calculate the new transition matrix
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

    return np.sum(xi_dist, 2) / np.sum(gamma_dist, axis=1).reshape((-1, 1))


# Calculate the new emission matrix
def new_emissions(alp, bet):
    numerators = np.ones((N, K, T))
    denominators = np.ones((N, K, T))

    for state in range(N):
        for symbol in range(K):

            for t in range(0, T):
                gamma_t = gamma(state, t, alp, bet)
                numerators[state][symbol][t] = gamma_t * int(X[t] == symbol)
                denominators[state][symbol][t] = gamma_t

    return np.sum(numerators, 2) / np.sum(denominators, 2)


# Forwards step to generate alpha distribution
def forwards():
    a = np.zeros((N, T))
    scale = np.ones(T)
    for position in range(T):
        for state in range(N):
            a[state][position] = alpha(state, position, a)
        scaling_factor = a[:, position].sum()
        scale[position] = scaling_factor
        a[:, position] = a[:, position] / scaling_factor

    return a, scale


# Backwards step to generate beta distribution
def backwards(scale):
    b = np.zeros((N, T))
    for position in range(1, T + 1):
        for state in range(N):
            b[state][-position] = beta(state, -position, b) / scale[-position]

    return b


# INIT function:
# Initialise parameters, variables, and formulate sequence into list of ints
def init(no_states, sequence, sigma):

    # Formulate observations `X` as list of ints
    obs = []
    contents = {}
    count = 0

    for l in sequence:
        if str(l) in contents.keys():
            obs.append(contents[str(l)])
        else:
            obs.append(count)
            contents.update({str(l): count})
            count += 1

    # Size of alphabet `K`
    sequence_symbs = len(list(contents.values()))
    no_symbs = len(sigma)

    if no_symbs < sequence_symbs:
        raise RuntimeError('Inputted alphabet `alphabet` must (at least) contain all observations seen in the observation sequence:'
                           f'\n     No. obs in sequenece = {sequence_symbs}, no. obs in alphabet = {no_symbs}')

    # Number of observation positions `T`
    positions = len(obs)

    # Transition matrix `M`
    M = np.full((no_states, no_states), 1/no_states)

    # Emission matrix `E`
    E = []
    for state in range(1, no_states + 1):
        em = np.array([symb for symb in range(state, no_states * no_symbs + 1, no_states)])
        E.append(np.divide(em, em.sum()))
    E = np.array(E)

    # Initial matrix `I`
    I = np.full(no_states, 1/no_states)

    return obs, no_symbs, positions, M, E, I


# CONVERGENCE function
# Check whether the parameters of the HMM have converged (changed less than `epsilon` at last update)
def convergence(nt, ne, ni, epsilon=0.00001):
    del_t = np.max(np.abs(transitions - nt))
    del_e = np.max(np.abs(emissions - ne))
    del_i = np.max(np.abs(initials - ni))

    return (del_t < epsilon) and (del_e < epsilon) and (del_i < epsilon)


# CALL THIS FUNCTION TO CONDUCT EM (BAUM-WELCH)
# RUN function combining expectation and maximisation step
def run(observations, no_states, alphabet):

    if type(no_states) is not int:
        raise TypeError('Inputted number of states `k` must be of type `int`')
    if type(observations) is not str and type(observations) is not list and type(observations) is not tuple:
        raise TypeError('Inputted observations sequence `observations` must be of type `str`, `list`, or `tuple`')

    # GLOBALS:
    global N  # `N` is the number of possible states
    global X  # `X` is the observation sequence
    global K  # `K` is the number of symbols in the alphabet
    global T  # `T` is the number of positions in `X` (`t` used to refer to a position 0 <= `t` < `T`)
    global transitions
    global emissions
    global initials

    # Initialise variables
    N = no_states
    X, K, T, transitions, emissions, initials = init(N, observations, alphabet)
    converged = False
    count = 0

    # Update parameters of HMM until they have converged
    while not converged and count <5:
        count += 1
        # E-step: generate alpha/beta distributions
        alpha_dist, scaling = forwards()
        beta_dist = backwards(scaling)

        # M-step: update parameters using these distributions
        new_init = new_initials(alpha_dist, beta_dist)
        new_trans = new_transitions(alpha_dist, beta_dist)
        new_emiss = new_emissions(alpha_dist, beta_dist)

        # Check convergence of parameters
        converged = convergence(new_trans, new_emiss, new_init)

        # Update global parameters to new values
        initials, transitions, emissions = new_init, new_trans, new_emiss

    # We know the first 5 decimal places of the parameters have converged (as convergence threshold `epsilon` = 0.00001)
    # Thus, round to 5 decimal places
    transitions, emissions, initials = np.round(transitions, decimals=5), np.round(emissions, decimals=5), np.round(initials, decimals=5)

    return transitions, emissions, initials

