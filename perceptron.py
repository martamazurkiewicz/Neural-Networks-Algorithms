#!/usr/bin/env python3

import matplotlib.pyplot as plt
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

def calcute_new_weights_directly(old_weights_vector, ro,input_vector,desired_val,output_val):
    return old_weights_vector + ro*(desired_val-output_val)*input_vector


def pa(train_set, teacher_set, init_weights, ro, activation_function, max_iterations = 10000):
    if not all(len(l) == len(train_set[0]) for l in iter(train_set)):
        raise Exception("Vectors in training set X must be of same length.")
    if(len(train_set) != len(teacher_set)):
        raise Exception("Each vector in training set must have a corresponding value in teacher set.")
    if(not isinstance(max_iterations, int) or max_iterations <= 0):
        raise Exception("Maximum number of iterations must be an integer greater than 0.")
    if((not isinstance(ro, int) and not isinstance(ro, float)) or ro <= 0 or ro > 1):
        raise Exception("The learning rate must be a number between 0(exclusive) and 1(inclusive)")
    weights_vector = init_weights
    iterations = 0
    weights_plot = np.empty((0, 3), dtype=float)
    while iterations < max_iterations:
        iterations += 1
        weights_plot = np.append(weights_plot, np.array([weights_vector]), axis=0)
        for i in range(len(train_set)):
            fi = weighted_sum(train_set[i], weights_vector)
            y = activation_function(fi)
            if y==teacher_set[i]:
                if i == len(train_set)-1 and testing_weights:
                    return (True, weights_vector, iterations, weights_plot)
                else:
                    testing_weights = True
            else:
                testing_weights = False
                weights_vector = calcute_new_weights_directly(weights_vector, ro,train_set[i],teacher_set[i],y)
                break
    return (False, None, iterations, None)

def bupa(train_set, teacher_set, init_weights, ro, activation_function, max_iterations = 10000):
    if not all(len(l) == len(train_set[0]) for l in iter(train_set)):
        raise Exception("Vectors in training set X must be of same length.")
    if(len(train_set) != len(teacher_set)):
        raise Exception("Each vector in training set must have a corresponding value in teacher set.")
    if(not isinstance(max_iterations, int) or max_iterations <= 0):
        raise Exception("Maximum number of iterations must be an integer greater than 0.")
    if((not isinstance(ro, int) and not isinstance(ro, float)) or ro <= 0 or ro > 1):
        raise Exception("The learning rate must be a number between 0(exclusive) and 1(inclusive)")
    weights_vector = init_weights
    misclassifications = 0
    iterations = 0
    weights_plot = np.empty((0, 3), dtype=float)
    while iterations < max_iterations:
        iterations += 1
        weights_plot = np.append(weights_plot, np.array([weights_vector]), axis=0)
        misclassifications = 0
        misclassified_vectors = []
        for i in range(len(train_set)):
            fi = weighted_sum(train_set[i], weights_vector)
            y = activation_function(fi)
            if y!=teacher_set[i]:
                misclassified_vectors.append((teacher_set[i] - y) * train_set[i])
                misclassifications += 1
        if len(misclassified_vectors) > 0:
            weights_vector = calculate_new_weights(misclassified_vectors, weights_vector, ro)
        else:
            break
    if(misclassifications == 0):
        return (True, weights_vector, iterations, weights_plot)
    else:
        return (False, None, iterations, None)

def show_plot(weights_plot_pa, weights_plot_bupa):
    fig, (ax0, ax1) = plt.subplots(2, 1, constrained_layout=True, figsize=(9, 8))
    x = np.linspace(-5,5,100)
    color = plt.cm.get_cmap('hsv', len(weights_plot_pa))
    ax0.set_title('Decision boundaries - PA')
    for i in range(len(weights_plot_pa)):
        [w0, w1, w2] = weights_plot_pa[i]
        if w2 != 0:
            y=-w0/w2-w1/w2*x
            ax0.plot(x,y,c=color(i),label='Iteration '+ str(i+1))
            ax0.legend(loc='upper right')
    ax0.scatter([0,0,1,1],[0,1,0,1],color='black')
    ax0.grid()
    ax0.set_ylabel('y')
    ax0.set_xlabel('x')
    color = plt.cm.get_cmap('hsv', len(weights_plot_bupa))
    ax1.set_title('Decision boundaries - BUPA')
    for i in range(len(weights_plot_bupa)):
        [w0, w1, w2] = weights_plot_bupa[i]
        if w2 != 0:
            y=-w0/w2-w1/w2*x
            ax1.plot(x,y,c=color(i),label='Iteration '+ str(i+1))
            ax1.legend(loc='upper right')
    ax1.scatter([0,0,1,1],[0,1,0,1],color='black')
    ax1.grid()
    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    plt.show(block=True)

def main():
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
    EXP_RES_VECT = np.array(D)
    ### END CONFIGURATION ###

    has_converged_pa, weights, iterations, weights_plot_pa = pa(X_VECT, EXP_RES_VECT, INIT_WEIGHTS, RO, ACTIVATION_FN)
    print("Results for PA perceptron:")
    if has_converged_pa:
        print(f"Final weights after {iterations} iterations: {weights}")
    else:
        print("The algorithm did not converge")
    has_converged_bupa, weights, iterations, weights_plot_bupa = bupa(X_VECT, EXP_RES_VECT, INIT_WEIGHTS, RO, ACTIVATION_FN)
    print("Results for BUPA perceptron:")
    if has_converged_bupa:
        print(f"Final weights after {iterations} iterations: {weights}")
    else:
        print("The algorithm did not converge")
    if has_converged_pa and has_converged_bupa:
        show_plot(weights_plot_pa, weights_plot_bupa)

if __name__ == "__main__":
    main()
