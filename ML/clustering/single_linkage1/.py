from lib1309829 import *

import sys


string = (sys.argv)

files = string[1::]


v = []

for f in files:

    v.append(preprocess(f,10))


s = []


for i in v:

    s.append(len(i))

s = sorted(s)
print(s[::-1])

    
D = DistanceMatrix(v)
np.set_printoptions(precision=2)
print(D)

print(D.mean())

res = single_linkage(D,3,1)
print(res)
