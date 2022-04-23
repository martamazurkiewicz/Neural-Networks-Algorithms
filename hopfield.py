#!/usr/bin/env python3

import numpy as np

def is_symmetric_matrix(numpy_array):
    transposed_matrix = numpy_array.transpose()
    if numpy_array.shape == transposed_matrix.shape:
        if (numpy_array == transposed_matrix).all():
            return True
    return False

def has_non_negative_diagonal(numpy_array):
    shape = numpy_array.shape
    if shape[0] != shape[1]:
        raise ValueError("Array passed to has_non_negative_diagonal needs to be a square matrix")
    for i in range(shape[0]):
        if numpy_array[i][i] < 0:
            return False
    return True

def is_positive_definite_matrix(numpy_array):
    if not is_symmetric_matrix(numpy_array):
        return False
    try:
        cholesky_dec = np.linalg.cholesky(numpy_array)
    except np.linalg.LinAlgError:
        return False
    return True

def u_vector(weights_array, bias, in_vector):
    if in_vector.transpose().shape != (weights_array.shape[0],):
        raise ValueError("Shapes of weights_array and in_vector do not match.")
    ws = weights_array.dot(in_vector)
    for i in range(len(in_vector)):
        ws[i] -= bias[0][i]
    return ws

def v_vector(u_vector, activation_fn):
    u_shape = u_vector.shape
    v_vector = np.zeros((u_shape[0],))
    for i in range(u_shape[0]):
        v_vector[i] = activation_fn(u_vector[i])
    return v_vector

def net_init_vectors(vect_dim, classes):
    if len(classes) != 2:
        raise ValueError("Argument classes must be a list containing 2 classes labels (integers)")
    count = 2**vect_dim
    vectors_array = np.zeros((count, vect_dim))
    n = vect_dim
    for i in range(count):
        binary_row = bin(i)[2:].zfill(n)
        binary_list = list(classes[int(x)] for x in binary_row)
        for j in range(vect_dim):
            vectors_array[i][j] = binary_list[j]
    return vectors_array

def energy(weights, biases, x_t_min_1, x_t):
    transposed_weights = weights.transpose()
    if not weights.shape == transposed_weights.shape:
        raise ValueError("Weights must be a square matrix.")
    if not len(biases[0]) == len(weights[0]):
        raise ValueError("Biases array must be of same length as length of weights array row")
    sum_1 = 0
    sum_2 = 0
    for i in range(weights.shape[0]):
        for j in range(weights.shape[0]):
            sum_1 += weights[i][j]*x_t[i]*x_t_min_1[j]
        sum_2 += biases[0][i] * (x_t[i] * x_t_min_1[i])
    e = -sum_1 + sum_2
    return e

def hopfield(WEIGHTS_ARRAY, B_ARRAY, ACTIVATION_FN, OBS_CLASSES):
    is_symmetric = is_symmetric_matrix(WEIGHTS_ARRAY)
    non_negative_diagonal = has_non_negative_diagonal(WEIGHTS_ARRAY)
    is_positive_definite = is_positive_definite_matrix(WEIGHTS_ARRAY)
    print(f"a) Is weights matrix symmetric: {'[Yes]' if is_symmetric else '[No]' }")
    print(f"b) Does weights matrix have non negative values on diagonal: {'[Yes]' if non_negative_diagonal else '[No]' }")
    print(f"c) Is weights matrix positive definite: {'[Yes]' if is_positive_definite else '[No]' }")
    print(f"\nIs the stabilization criterion satisfied: {'[Yes]' if (is_symmetric and non_negative_diagonal and is_positive_definite) else '[No]' }\n")

    INIT_VECTORS = np.array(net_init_vectors(WEIGHTS_ARRAY.shape[1], OBS_CLASSES))
    for vector in INIT_VECTORS:
        u = u_vector(WEIGHTS_ARRAY, B_ARRAY, vector)
        v = v_vector(u, ACTIVATION_FN)
        if all(v == vector):
            print(f"[#] Net point {vector} is a network FIXED POINT.\n")
            continue
        results = []
        results.append(vector)
        results.append(v)
        while True:
            old_x = v
            u = u_vector(WEIGHTS_ARRAY, B_ARRAY, v)
            v = v_vector(u, ACTIVATION_FN)
            results.append(v)
            if (old_x == v).all():
                break
            if all(v == vector):
                break
        if all(old_x == v):
            print(f"[#] Net point {vector} stabilized on net fixed point {v}\n")
        elif all(v == vector):
            print(f"[#] Net cycle detected for vector: {vector}\n")
            print(f" - Cycle: {results[:-1]} (...)")
            cycle_length = 1
            i = 1
            while i < len(results) and not all(results[i] == results[0]):
                cycle_length += 1
                i += 1
            print(f" - Cycle length: {cycle_length}\n")
        else:
            print(f"CRITICAL ERROR. VECTOR {vector} left without result.")

def main():
    ACTIVATION_FN = lambda x: 1 if x> 0 else -1
    OBS_CLASSES = [1, -1]
    weights_1 = np.array([
        [0, -2/3, 2/3],
        [-2/3, 0, -2/3],
        [2/3, -2/3, 0]
    ])
    biases_1 = np.array([
        [0, 0, 0]
    ])
    print("###############")
    print("ZADANIE 1")
    print("###############")
    print(f"Weights:\n{weights_1}")
    print(f"Biases:\n{biases_1}")
    hopfield(weights_1, biases_1, ACTIVATION_FN, OBS_CLASSES)
    weights_2 = np.array([
        [0, 1],
        [-1, 0]
    ])
    biases_2 = np.array([
        [0, 0]
    ])
    print("###############")
    print("ZADANIE 2")
    print("###############")
    print(f"Weights:\n{weights_2}")
    print(f"Biases:\n{biases_2}")
    hopfield(weights_2, biases_2, ACTIVATION_FN, OBS_CLASSES)

if __name__ == '__main__':
    main()
