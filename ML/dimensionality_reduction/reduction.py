import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
from lib_reduction import *


start = time.time()

string = sys.argv


file = string[1]

reduction = [int(i) for i in string[2:]]


frame = pd.read_csv(file)


frameP = frame.pivot(index="userId", columns="movieId", values="rating")

frameP = frameP.fillna(frameP.mean(axis=0), axis=0, inplace=True)

X = frameP.values

X_r = []
disto = []


lim = min(250, X.shape[0])

dist = alldist(X[:lim])


for k in reduction:

    r = reduce(X, k)

    X_r.append(r)

    dist_r = alldist(r[:lim])

    disto.append(distortion(dist, dist_r))


tot = frameP.memory_usage(index=True).sum()


for j in range(len(X_r)):

    print(two(tot / (X_r[j].nbytes)), two(disto[j].min()), two(disto[j].mean()), two(disto[j].max()))


for i in range(len(reduction)):

    if reduction[i] == min(reduction):

        g = i

r_min = min(disto[g])
r_max = max(disto[g])


for i in range(len(disto)):

    plt.hist(disto[i], bins=60, range=(r_min, r_max), histtype="step", label="d = %d" % reduction[i], normed=True)


end = time.time()

# print(end-start)

plt.xlabel("distortion")
plt.ylabel("frequency")
plt.legend()
plt.grid()
plt.show(block=True)
# plt.close("all")
