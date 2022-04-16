#!/usr/bin/env python3

import numpy as np
from bupa import bupa

def calc_rbf_value(vector):
    return np.exp(-(np.linalg.norm(vector[1]-vector[2])**2)/2)

def main():
    ### CONFIGURATION ###
    X = [[1,0,0], [1,0,1], [1,1,0], [1,1,1]]
    D_XOR = [0, 1, 1, 0]
    RO = 1
    ACTIVATION_FN = lambda x: 1 if x > 0 else 0
    INIT_WEIGHTS = np.array([0.5, 0, 1, 0])
    X_VECT = np.array(X)
    MAX_ITERATIONS = 10000
    EXP_RES_VECT = np.array(D_XOR)
    ### END CONFIGURATION ###
    kernel_values = np.zeros(len(X_VECT))
    for i in range(len(X_VECT)):
        kernel_values[i] = calc_rbf_value(X_VECT[i])
    training_set = np.c_[X_VECT, kernel_values]

    has_converged, weights, iterations = bupa(training_set, EXP_RES_VECT, INIT_WEIGHTS, RO, ACTIVATION_FN)
    if has_converged:
        print(f"Final weights after {iterations} iterations: {weights}")
    else:
        print("The algorithm did not converge")

if __name__ == '__main__':
    main()