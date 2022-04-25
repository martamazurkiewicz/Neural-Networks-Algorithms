from math import e
import numpy as np


def dot_product(x,w):
    fi = 0
    for i in range(len(x)):
        fi += x[i]*w[i]
    return fi

def f(w,x):
    return 1/(1+e**(-dot_product(x,w)))

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

def is_gradient_changing(weights_diff):
    for i in range(len(weights_diff)): 
        for j in range(len(weights_diff[i])):
            if abs(weights_diff[i,j]) > 0.0005:
                return True
    return False

def update_weight(weights,ni,energy_gradient):
    updated_weights = get_updated_weights(weights,ni,energy_gradient)
    weights_diff = updated_weights-weights
    weights = updated_weights
    return [weights, weights_diff]

def network_partial_energy(weights,input_vector,d,ni):
    energy = np.ones(4)
    weights_diff = np.ones([3,3])
    iteration = 0
    while is_gradient_changing(weights_diff) and iteration < 8000:
        for i in range(len(input_vector)):
            [partial_energy, output_vectors] = cycle(weights,input_vector[i],d[i])
            energy_gradient = gradient_matrix(d[i],weights[2],output_vectors)
            energy[i] = partial_energy
            [weights,weights_diff] = update_weight(weights,ni,energy_gradient)
        iteration += 1
    print('Partial energy method')
    print ('Iteration: ', iteration)
    print ('Last energy: ', energy)
    print ('Ending weights: ', weights)


def network_whole_energy(weights,input_vector,d,ni):
    energy = np.ones(4)
    weights_diff = np.ones([3,3])
    iteration = 0
    while is_gradient_changing(weights_diff) and iteration < 8000:
        energy_gradient_sum = np.zeros([3,3])
        for i in range(len(energy)):
            [partial_energy, output_vectors] = cycle(weights,input_vector[i],d[i])
            energy_gradient_sum += gradient_matrix(d[i],weights[2],output_vectors)
            energy[i] = partial_energy
        [weights,weights_diff] = update_weight(weights,ni,energy_gradient_sum)
        iteration += 1
    print('Whole energy method')
    print ('Iteration: ', iteration)
    print ('Last energy: ', energy)
    print ('Ending weights: ', weights)


def __main__():
    weights_init = np.array([[0.86,-0.16,0.28],[0.82,-0.51,-0.89],[0.04,-0.43,0.48]], np.double)
    x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]], np.double)
    d = [0,1,1,0]
    network_partial_energy(weights_init,x,d,0.5)
    network_whole_energy(weights_init,x,d,0.5)


# Partial energy method
# Iteration:  4473
# Last energy:  [0.00029608 0.00033231 0.00033374 0.00051152]
# Ending weights:  [[ 7.06050946 -4.7434732  -4.74838189]
#  [ 2.63533557 -6.44750261 -6.48129802]
#  [-4.54284054  9.59884354 -9.74491854]]
# Whole energy method
# Iteration:  3476
# Last energy:  [0.25056427 0.25220448 0.00084742 0.00112414]
# Ending weights:  [[ 6.95371339 -8.894576    5.744297  ]
#  [ 6.4696388  -6.5428181  -0.36700982]
#  [ 1.64809785 -7.44353026  5.79955897]]
