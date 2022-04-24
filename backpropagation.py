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


def updated_weights_partial(w,alpha,d,x3,x2,x1):
    updated_weights = np.empty_like(w)
    energy_gradient = gradient_matrix(d,w[2],x3,x2,x1)
    for i in range(len(w)):
        for j in range(len(w[i])):
            updated_weights[i,j] = w[i,j]-alpha*energy_gradient[i,j]
    return updated_weights

def cycle(w,x,d):
    w21=w[0]
    w22=w[1]
    w31=w[2]
    x1=[1,0,0]
    x2=[1,f(w21,x1),f(w22,x1)]
    x3=f(w31,x2)
    energy = get_energy(w21,w22,w31,x,d)
    print(energy)
    w = updated_weights_partial(w,1,d,x3,x2,x1)
    print(w)

def __main__():
    weights_init = np.array([[0.86,-0.16,0.28],[0.82,-0.51,-0.89],[0.04,-0.43,0.48]], np.double)
    x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]], np.double)
    d = [0,1,1,0]
    cycle(weights_init,x[0],d[0])
