import numpy as np
import matplotlib.pyplot as plt


def inner(X,theta):
    
    prod = ((X*theta).sum())
    
    return prod


def h_function(X,theta):
    
    m,p = X.shape
    h = np.zeros((m,))  
    
    for i in range(m):
    
        h[i] = inner(X[i,:],theta)
        
        
    return h

def Cost(X,theta,y):
    
    J = (h_function(X,theta)-y)**2
    
    J = J.sum()
    
    return J


def gradient(X,theta,y):
    
    '''
    y length m
    X m by p
    theta length p
    '''
    m,p = X.shape
    
    grad = np.zeros((p,))
    
    f = (h_function(X,theta)-y)
   
    for j in range(p):
        
        grad[j] = (2*(f*X[:,j])).sum()
        
                                  
    return grad

def descent(y, X, alpha = 1e-3, itr = 1e2, eps = 1e-6):
    
    m,p = X.shape
    itr = int(itr)
    theta = np.ones((p,))
    cost = np.zeros((itr,))
    n = np.zeros((itr,), dtype = int)


    for i in range(itr):

        temp = theta
        theta = theta - alpha*gradient(X,theta,y)

        err = np.linalg.norm(theta-temp)* 1/np.linalg.norm(temp)

        if err < eps:

            break


    return theta


def r2(y, theta, X):
    
    err_res = (((theta*X).sum(axis=1)-y)**2).sum()

    y_mean = np.mean(y)
    
    err_tot = ((y-y_mean)**2).sum()
    
    R = 1-(err_res/err_tot)
    
    return R


def normalization(X):
    
    n,p = X.shape
        
    m = np.mean(X,axis = 0)
    sigma = np.std(X,axis = 0)
    
    X_norm = np.ones((n,p+1))
    
    for i in range(p):
        
        X_norm[:,i+1] = (X[:,i] - m[i])/sigma[i]
    
    
    return X_norm

