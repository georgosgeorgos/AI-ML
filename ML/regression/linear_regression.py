import random
import csv
import sys
import numpy as np


class LinearRegression:
	
	def __init__(self,x_train,y_train):
		
		self.theta = np.zeros(x_train.shape[1]+1)
		self.x_train = np.ones((x_train.shape[0],x_train.shape[1]+1))
		self.x_train[:,1:] = x_train
		self.y_train = y_train
		self.N = x_train.shape[0]
		
	def h(self):
		return (self.theta*self.x_train).sum(axis=1)
	
	def cost(self):
		cost = ((self.h()-self.y_train)**2).sum()
		cost = cost/(2*self.N)
		return cost
	
	def gradient(self):
		gradient = ((self.h()-self.y_train)*self.x_train.T).sum(axis=1)
		gradient = gradient/self.N
		return gradient
	
	def standardization(self):
		self.x_train[:,1:] = (self.x_train[:,1:]-self.x_train[:,1:].mean(axis=0))/self.x_train[:,1:].std(axis=0)
	
	def fit(self,n,alpha):
		for i in range(n):
			self.theta -= alpha*self.gradient()   #descent direction
	
	def getParameters(self):
		return self.theta



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


N = 100
alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]


with open(name_output, 'w', newline='') as fp:

	for alpha in alphas:
		
		lr = LinearRegression(x_train,y_train)
		lr.standardization()
		lr.fit(N,alpha)
		params = lr.getParameters()

		fp.write(str(alpha) + "," + str(N) + "," + str(params[0]) + "," + str(params[1]) + "," + str(params[2]) + "\n")
		
	lr = LinearRegression(x_train,y_train)
	lr.standardization()
	lr.fit(10000,0.01)
	params = lr.getParameters()
	fp.write(str(0.01) + "," + str(100000) + "," + str(params[0]) + "," + str(params[1]) + "," + str(params[2]))




	