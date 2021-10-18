import numpy as np
import math
from time import time

loop = 10000
a = -100

stime = time()
for i in range(loop):
    #a = abs(a)
    a = abs(a)
print(time()-stime)
