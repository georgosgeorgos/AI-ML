import numpy as np


def charfreq(filename, filtr = "aeiou"):
    
    d = {c:0 for c in filtr}
    
    file = open(filename,'r')
    
    for line in file:
        
        for ch in line:
            
            if ch in d:
                
                d[ch] += 1
                
    file.close()
    
    v = np.zeros(len(filtr))
    
    for i in range(len(filtr)):
        
        v[i] = d[filtr[i]]
        
    return v/v.sum() if v.sum() > 0 else v


def euc(x,y):
    
    s = (x-y)**2
    e = np.sqrt(s.sum())
    
    return e

def cldist(c1,c2):
    
    dist = np.infty
    
    for i in range(len(c1)):
        for j in range(len(c2)):
            
            dist = min(dist,euc(c1[i], c2[j]))
    
    return dist



def closest(L):
    
    d = np.infty
    a,b = -1,-1
    
    for i in range(len(L)-1):
        for j in range(i+1,len(L)):
            
            t = cldist(L[i],L[j])
            
            if t < d:
                
                d = t
                a,b = i,j

    return a,b



def single_linkage(D, k):

    
    if type(D[0]) == list:
    
        DD = D.copy()
    
    else:
        
        DD = [[d] for d in D]
    
    C = [[kk] for kk in range(len(DD))]
    
    while len(C) > k:
        
        i, j = closest(DD)
        
        C[i].extend(C[j])
        
        C.remove(C[j])
        
        
        del DD[j]
        
        
    return C
        
       
