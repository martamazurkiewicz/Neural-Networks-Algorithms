#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def dot_product(x,w):
    return sum([xi * wi for (xi, wi) in zip(x,w)])

def f(w,x):
    return 1/(1+np.exp(-dot_product(x,w)))

def energy(d,y):
    return (d-y)**2

def get_x2(weights,x1):
    return [1,f(weights[0],x1),f(weights[1],x1)]

def get_x3(weights,x2):
    return f(weights[2],x2)

def gradient_matrix(d,w31,output_vectors):
    [x3,x2,x1]=output_vectors
    e_deriv = 2*(x3-d)
    x3_deriv = e_deriv*x3*(1-x3)
    x22_deriv = x3_deriv*x2[2]*(1-x2[2])*w31[2]
    x21_deriv = x3_deriv*x2[1]*(1-x2[1])*w31[1]
    return np.array([[x21_deriv*x1[0], x21_deriv*x1[1], x21_deriv*x1[2]], [x22_deriv*x1[0], x22_deriv*x1[1], x22_deriv*x1[2]], [x3_deriv*x2[0], x3_deriv*x2[1], x3_deriv*x2[2]]], np.double)


def get_updated_weights(weights,ni,energy_gradient):
    updated_weights = np.empty_like(weights)
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            updated_weights[i,j] = weights[i,j]-ni*energy_gradient[i,j]
    return updated_weights

def cycle(w,x1,d):
    x2 = get_x2(w,x1)
    x3 = get_x3(w,x2)
    return [energy(x3,d), [x3,x2,[x1[0],x1[1],x1[2]]]]

def is_gradient_changing(weights_diff,diff):
    for i in range(len(weights_diff)): 
        for j in range(len(weights_diff[i])):
            if abs(weights_diff[i,j]) > diff:
                return True
    return False

def update_weight(weights,ni,energy_gradient):
    updated_weights = get_updated_weights(weights,ni,energy_gradient)
    weights_diff = updated_weights-weights
    weights = updated_weights
    return [weights, weights_diff]

def network_partial_energy(weights,input_vector,d,ni,diff):
    solution = np.ones(4)
    energy = np.ones(4)
    weights_diff = np.ones([3,3])
    iteration = 0
    energies=np.empty((0,4), dtype=float)
    while is_gradient_changing(weights_diff,diff) and iteration < 8000:
        for i in range(len(input_vector)):
            [partial_energy, output_vectors] = cycle(weights,input_vector[i],d[i])
            solution[i] = output_vectors[0]
            energy_gradient = gradient_matrix(d[i],weights[2],output_vectors)
            energy[i] = partial_energy
            [weights,weights_diff] = update_weight(weights,ni,energy_gradient)
        iteration += 1
        energies = np.append(energies, np.array([energy]), axis=0)
    print ('Partial energy method')
    print ('Iteration: ', iteration)
    print ('Last energy: ', energy)
    print ('Solution: ', solution)
    print ('Ending weights: ', weights)
    return energies


def network_whole_energy(weights,input_vector,d,ni,diff):
    solution = np.ones(4)
    energy = np.ones(4)
    weights_diff = np.ones([3,3])
    iteration = 0
    while is_gradient_changing(weights_diff,diff) and iteration < 8000:
        energy_gradient_sum = np.zeros([3,3])
        for i in range(len(input_vector)):
            [partial_energy, output_vectors] = cycle(weights,input_vector[i],d[i])
            solution[i] = output_vectors[0]
            energy_gradient_sum += gradient_matrix(d[i],weights[2],output_vectors)
            energy[i] = partial_energy    
        [weights,weights_diff] = update_weight(weights,ni,energy_gradient_sum)
        iteration += 1
    print ('Whole energy method')
    print ('Iteration: ', iteration)
    print ('Last energy: ', energy)
    print ('Solution: ', solution)
    print ('Ending weights: ', weights)



def __main__():
    weights_init = np.array([[0.86,-0.16,0.28],[0.82,-0.51,-0.89],[0.04,-0.43,0.48]], np.double)
    x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]], np.double)
    d = [0,1,1,0]
    energies = network_partial_energy(weights_init,x,d,0.5,0.0005)
    #fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    #fig, ax0 = plt.subplots(constrained_layout=True,figsize=(7, 4))
    iter = range(len(energies))
    fig, ax0 = plt.subplots(constrained_layout=True,figsize=(8, 4))
    ax0.plot(iter, energies[:,0], label='E for vector [1,0,0]')
    ax0.plot(iter, energies[:,1], label='E for vector [1,0,1]')
    ax0.plot(iter, energies[:,2], label='E for vector [1,1,0]')
    ax0.plot(iter, energies[:,3], label='E for vector [1,1,1]')
    ax0.legend(loc='upper right')
    ax0.set_ylabel('Energy')
    ax0.set_xlabel('Iteration')
    plt.title('Energy change in partial energy methon')
    plt.show(block=True)
    #network_whole_energy(weights_init,x,d,0.5,0.000000001)