from math import e
import numpy as np


def dot_product(x,w):
    fi = 0
    for i in range(len(x)):
        fi += x[i]*w[i]
    return fi

def f(w,x):
    return 1/(1+e**(-dot_product(x,w)))

def L(d,y):
    return (d-y)**2

def get_energy(w21,w22,w31,x1,d):
    x2=[1,f(w21,x1),f(w22,x1)]
    x3=f(w31,x2)
    return L(d,x3)

def gradient_matrix(d,w31,x3,x2,x1):
    e_deriv = 2*(x3-d)
    x3_deriv = e_deriv*x3*(1-x3)
    x22_deriv = x3_deriv*x2[2]*(1-x2[2])*w31[2]
    x21_deriv = x3_deriv*x2[1]*(1-x2[1])*w31[1]
    return np.array([[x21_deriv*x1[0], x21_deriv*x1[1], x21_deriv*x1[2]], [x22_deriv*x1[0], x22_deriv*x1[1], x22_deriv*x1[2]], [x3_deriv*x2[0], x3_deriv*x2[1], x3_deriv*x2[2]]], np.double)


def updated_weights_partial(w,ni,d,x3,x2,x1):
    updated_weights = np.empty_like(w)
    energy_gradient = gradient_matrix(d,w[2],x3,x2,x1)
    for i in range(len(w)):
        for j in range(len(w[i])):
            updated_weights[i,j] = w[i,j]-ni*energy_gradient[i,j]
    return updated_weights

def cycle(w,x,d):
    w21=w[0]
    w22=w[1]
    w31=w[2]
    x1=x
    x2=[1,f(w21,x1),f(w22,x1)]
    x3=f(w31,x2)
    energy = get_energy(w21,w22,w31,x,d)
    w = updated_weights_partial(w,0.5,d,x3,x2,x1)
    return [energy, w]

def is_energy_bigger(w_diff):
    for i in range(len(w_diff)): 
        for j in range(len(w_diff[i])):
            if w_diff[i,j] > 0.001:
                return True
    return False

def network(w,x,d):
    energy = np.ones(4)
    weights_diff = np.ones([4,4])
    iteration = 0
    while is_energy_bigger(weights_diff) and iteration < 15000:
        for i in range(len(energy)):
            [partial_energy, updated_weights] = cycle(w,x[i],d[i])
            weights_diff=abs(updated_weights-w)
            w = updated_weights
            energy[i] = partial_energy
            iteration += 1
    print ('Iteration: ', iteration)
    print ('Last energy: ', energy)
    print ('Ending weights: ', w)

def __main__():
    weights_init = np.array([[0.86,-0.16,0.28],[0.82,-0.51,-0.89],[0.04,-0.43,0.48]], np.double)
    x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]], np.double)
    d = [0,1,1,0]
    network(weights_init,x,d)


# Iteration:  10808
# Last energy:  [0.00057089 0.00065929 0.00066268 0.00103289]
# Ending weights:  [[ 6.67503168 -4.49533433 -4.49991181]
#  [ 2.52301385 -6.28410326 -6.32430058]
#  [-4.20883312  8.95986141 -9.12781289]]
