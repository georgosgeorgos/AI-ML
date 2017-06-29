# final

import random
import numpy as np
import re


def wordcount(book):
    
    wc = {}
    
    for line in book:
        
        line = re.compile('\w+').findall(line)
        
        for word in line:
            
            wc[word.lower()] = wc.get(word.lower(),0) + 1
                
    
    return wc


def bag(wc, threshold=1):
    
    bag = []
    
    for key in wc:
        
        if wc[key] >= threshold:
            
            bag.append(key)
    bag = set(bag)
    
    return bag
            

    
def jaccard(s1, s2):
    
    num = s1.intersection(s2)
    den = s1.union(s2)
    
    J = len(num)/len(den)
    
    return J



def preprocess(f, threshold = 10):
    
    book = open(f,"r")
    wc = wordcount(book)
    new = bag(wc,threshold)
    book.close()
    return new



def DistanceMatrix(v):
    
    l = len(v)
    D = np.zeros((l,l))
    
    for i in range(l):
        for j in range(l):

            D[i,j] = 1-jaccard(v[i],v[j])
        
    return D


def single_linkage_extract(clusters,D):
    
    d = len(D)
    
    K = [i for i in range(d)]
    clu_num = [i for i in range(len(clusters))]
    
    for c in range(len(clusters)):
        
        for i in clusters[c]:
            
            K[i] = clu_num[c]
            
    return K





def single_linkage(D, k=2, t = 0):
    
    
    d = len(D)
    Dc = D.copy()
    diag = np.diag_indices(d)
    Dc[diag] = 1

    clusters = [[i] for i in range(d)]
    
    
    while len(clusters) > k:

        ii = np.where(Dc == Dc.min())
        
        
        if len(set(ii[0])) > 1:

            r = random.randint(0,len(set(ii[0]))-1)
            i  = ii[0][r]
            j  = ii[1][r]
            
        else:
            i  = ii[0][0]
            j  = ii[1][0]

            
        clusters[i].extend(clusters[j])   
        
    
        del clusters[j]
        

        for g in range(len(clusters)):

            if Dc[i,g] > Dc[j,g]:

                Dc[i,g] = Dc[j,g] 
                Dc[g,i] = Dc[j,g]

        Dc[i,i] = 1
        Dc = np.delete(Dc,j,1)
        Dc = np.delete(Dc,j,0) 
    
    if t == 0:
        
        return single_linkage_extract(clusters,D)
    
    else:

        return clusters 





