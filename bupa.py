#!/usr/bin/env python3

import numpy as np

def weighted_sum(data_vector, weights_vector):
    return data_vector.dot(weights_vector)

def calc_z_vector(misclassified_vectors):
    vect_length = len(misclassified_vectors[0])
    z_vect = [0 for _ in range(vect_length)]
    for vect in misclassified_vectors:
        z_vect += vect
    return z_vect

def calculate_new_weights(misclassified_vectors, old_weights_vector, ro):
    return old_weights_vector + ro * calc_z_vector(misclassified_vectors)

### CONFIGURATION ###
X = [[1,0,0], [1,0,1], [1,1,0], [1,1,1]]
D_AND = [0, 0, 0, 1]
D_XOR = [0, 1, 1, 0]
D_OR = [0, 1, 1, 1]
RO = 1
ACTIVATION_FN = lambda x: 1 if x > 0 else 0
INIT_WEIGHTS = np.array([0.5, 0, 1])
X_VECT = np.array(X)
MAX_ITERATIONS = 10000
D = D_AND
### END CONFIGURATION ###

EXP_RES_VECT = np.array(D)
misclassifications = 0
iterations = 0
weights_vector = INIT_WEIGHTS

if not all(len(l) == len(X[0]) for l in iter(X)):
    print("Vectors in training set X must be of same length")
    exit(1)
if(len(X) != len(D)):
    print("Each vector in training set X must have a corresponding value in desired values vector D")
    exit(2)

while iterations < MAX_ITERATIONS:
    weights_before_for_loop = weights_vector
    misclassifications = 0
    misclassified_vectors = []
    for i in range(len(X)):
        fi = weighted_sum(X_VECT[i], weights_vector)
        y = ACTIVATION_FN(fi)
        if y!=EXP_RES_VECT[i]:
            misclassified_vectors.append((EXP_RES_VECT[i] - y) * X_VECT[i])
            misclassifications += 1
    if len(misclassified_vectors) > 0:
        weights_vector = calculate_new_weights(misclassified_vectors, weights_before_for_loop, RO)
    if (weights_before_for_loop == weights_vector).all():
        break
    iterations += 1

if misclassifications == 0:
    print(f"Final weights: {weights_vector}")
else:
    print("The algorithm did not converge")
