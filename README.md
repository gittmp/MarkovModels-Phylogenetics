# MarkovModels-Phylogenetics

## Implemenations

This step of the project considers a Markov Model (MM) on a finite set of states *S* that generates a sequence of letters from a finite alphabet *Σ*. The parameters of the model are the initial probabilities $p(s), s \in S$, the transition matrix $m_st, s \in S,t \in S$, and the emission probabilities $e_s(a), s \in S, a \in Σ$.

The Expectation-Maximisation (EM) algorithm is implemented for estimating the parameters of the model, given a sequence of observed letters (but not the sequence of states that generated it) and assuming that the number of states "k" is also given.

## Written

1 
a) Develop an algorithm that takes a sequence of "n" letters as an input and outputs the most likely sequence of states that generated the input. Prove the correctness of your algorithm and estimate the running time and the space used in terms of "n" and the number of states of the model "k". 

2 
a) Explain the BUILD algorithm in [1] and how it works in your own words. Do not use pseudocode.

b) Expand the partition step (given below) in pseudocode:

$compute (\pi C) = S1, S2, ... Sr$

c) Run the algorithm.

d) “Reverse” the BUILD algorithm, i.e. design an algorithm that takes a tree with labeled leaves as an input, and produces a set of constraints of the form $(i, j) < (k, l)$, such that when BUILD runs on that set, the result is (an isomorphic copy of) the input tree. Prove the correctness of your algorithm.
