import importlib

lib = importlib.import_module("lib_wine_regression")


import matplotlib.pyplot as plt
import numpy as np
import csv


f = open("wine.csv", "r")
ff = csv.reader(f)

data = []
for i in ff:

    data.append(i)
data = np.array(data)


X = data[1::, [i for i in range(len(data[1, :])) if i != 1]]
X = np.array((X)).astype(float)

y = data[1:, 1]
y = np.array((y)).astype(float)


X_norm = lib.normalization(X)


X_one = np.ones((len(X[:, 0]), 2))
X_one[:, 1] = X_norm[:, 3]

theta_one = lib.descent(y, X_one, alpha=1e-3, itr=1e4)

R_one = lib.r2(y, theta_one, X_one)

print(theta_one)
print(R_one)

x = np.linspace(-3, 3, 100)
plt.plot(x, theta_one[0] + theta_one[1] * x)
plt.plot(X_one[:, 1], y, "or")
plt.title("The Wine Equation")
plt.ylabel("Price")
plt.xlabel("AGST")

plt.savefig("wine.png", bbox_inches="tight")


theta = lib.descent(y, X_norm, alpha=1e-3, itr=1e4)

R = lib.r2(y, theta, X_norm)

print(theta)
print(R)
