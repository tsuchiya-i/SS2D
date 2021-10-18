import math, time
import numpy as np

a = np.zeros((5, 5))

b = np.array([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])

c = np.array([1,1])

print(np.linalg.norm(c))

"""
count = 0
stime = time.time()
for k in range(100000):
    for i in range(5):
        for j in range(5):
            if b[i][j] > 0:
                count = count+1

ftime=time.time()-stime
print("ftime = "+str(ftime))
print(count)
"""
