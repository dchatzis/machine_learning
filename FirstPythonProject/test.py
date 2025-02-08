import math, copy
import numpy as np
import matplotlib.pyplot as plt

w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10,20,30])

f1 = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b

f2 = 0
for j in range(w.shape[0]):
    f2 = f2 + w[j]*x[j]
f2 = f2 + b

f3 = np.dot(w,x) + b

print(f"f1 is equal: {f1}")
print(f"f2 is equal: {f2}")
print(f"f3 is equal: {f3}")
