import numpy as np
import random



def euclidian(x,y):
    
    res = np.sqrt(((y-x)**2).sum())
    
    return res

def alldist(X):
    
    n,m = X.shape    
    dist = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i+1,n):
            
            dist[i,j] = euclidian(X[i,:],X[j,:])
            dist[j,i] = dist[i,j]
            
    return dist

def achmat(D,d):
    
    '''
    input: int D >> d
    output: matrix {-1,1}
    '''
    
    return np.sign((np.random.randn(D,d) < 0) - 0.5)


def reduce(X,d):
    
    n,D = X.shape
    
    A = achmat(D,d)
    
    res = np.zeros((n,d))
    
    res = np.dot(X,A)/np.sqrt(d)
    
    return res


def distortion(dist,dist_red):
    
    n = dist.shape[0]
    v1 = []
    v2 = []
    
    for i in range(n):
        for j in range(i+1,n):
            
            v1.append(dist[i,j])
            v2.append(dist_red[i,j])
    
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    res = v2/v1
    
    return res    
    
    
def control(X,res):
    
    n = X.shape[0]
    
    return len(res) == n*(n-1)/2
    
    

def two(f):

    return "%0.2f" % f
    
    
    
    