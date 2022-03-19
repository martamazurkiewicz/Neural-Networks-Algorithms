def dot_product(xi,w):
    fi = 0
    for i in range(len(xi)):
        fi += xi[i]*w[i]
    return fi

def y_func(x):
    if x > 0:
        return 1
    else:
        return 0
        
def weights(w,xi,ro,d,y):
    return tuple(w[i]+ro*(d-y)*xi[i] for i in range(len(xi)))
    

def perceptron():
    x=[(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    d=[0,0,0,1]
    w=(0.5,0,1)
    fi = 0
    ro = 1
    for i in range(50):
        w_temp = w
        for j in range(len(x)):
            fi = dot_product(x[j],w)
            y = y_func(fi)
            if y != d[j]:
                w = weights(w,x[j],ro,d[j],y)
                break
            elif j == len(x)-1 and w_temp == w:
                return w
            
def main():
    print(perceptron())

if __name__ == "__main__":
    main()