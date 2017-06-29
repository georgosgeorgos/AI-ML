import random
import csv
import sys
import numpy as np

class Perceptron:
    
    def __init__(self,x_train,y_train):
        
        self.theta = np.random.random(x_train.shape[1]+1)
        self.x_train = np.ones((x_train.shape[0],x_train.shape[1]+1))
        self.x_train[:,1:] = x_train
        self.y_train = y_train
        self.data = []
        
    def fit(self):
        c = -1
        while True and c != 0:
            
            res = (self.x_train*self.theta).sum(axis=1)

            res[res>0] = 1
            res[res<0] = -1
            
            c = 0
            for i in range(self.x_train.shape[0]):
                
                if self.y_train[i]*res[i] <= 0:
                    c +=1
                    self.theta += self.y_train[i]*self.x_train[i,:]
            
            self.data.append([int(self.theta[1]),int(self.theta[2]),int(self.theta[0])])
        
        return self.data
    
    def getParameters(self):
        return self.theta[1:]
    
    def predict(self):
        
        res = (self.x_train*self.theta).sum(axis=1)
        res[res>0] = 1
        res[res<0] = -1
        
        return res
    
    def accuracy(self):
        
        res = (self.x_train*self.theta).sum(axis=1)
        res[res>0] = 1
        res[res<0] = -1
        
        accuracy = ((res == self.y_train)*1).sum()/len(self.y_train)
        
        return accuracy



string =sys.argv

name_input = string[1]
name_output = string[2]

with open(name_input,"r") as f:
    
    ff = csv.reader(f)
    data = []
    for row in ff:
        data.append(row)

data = np.array(data).astype(float)

x_train = data[:,:-1]
y_train = data[:,-1]

perceptron = Perceptron(x_train,y_train)
res = perceptron.fit()
#print(perceptron.accuracy())
with open(name_output, 'w', newline='') as fp:
    
    out = csv.writer(fp, delimiter=',')
    out.writerows(res)
