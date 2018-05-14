from lib_clustering_freq import *

import sys
import numpy as np

string = (sys.argv)

files = string[1:len(string)-2]

filtr = string[-2]

k = int(string[-1])


files_dict = {i:v for i,v in enumerate(files)}


D = [charfreq(f, filtr) for f in files]


#np.set_printoptions(precision=2)


vector = single_linkage(D, k) 


pp = [[] for i in range(len(vector))]

for j in range(len(vector)):

	for i in range(len(vector[j])):
		
		pp[j].append(files_dict[vector[j][i]])


for p in pp:

	print(p)

