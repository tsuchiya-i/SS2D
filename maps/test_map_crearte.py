import numpy as np
import math
import cv2

height = 200
width = 400

testmap = np.zeros((height,width))

for i in range(height):
    for j in range(width):
        if i<1 or i==height-1 or j==0 or j>200:
            testmap[i][j] = 0
        else:
            testmap[i][j] = 255

cv2.imwrite('./test_map.jpg', testmap)
