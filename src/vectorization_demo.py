import numpy as np
import time

a = np.random.normal(0, 1, 1000000)
b = np.random.normal(0, 1, 1000000)

# For lop vector multiplication
t1 = time.time()
c = 0
for i in range(1, a.size):
    c += a[i] * b[i]

t2 = time.time()
print('Time taken in for loop calculation : ', ((t2 - t1) * 1000))

# numpy vector multiplication
t1 = time.time()
c = np.dot(a, b)
t2 = time.time()
print('Time taken in vector calculation : ', ((t2 - t1) * 1000))

