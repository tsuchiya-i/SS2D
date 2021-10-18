import numpy as np
import cv2

#img = cv2.imread('./paint_map/free_sim.png', cv2.IMREAD_COLOR)
img = cv2.imread('./nakano_11f_sim.png', cv2.IMREAD_COLOR)
window_name = 'img'
H, W = img.shape[:2]

print({"H":H,"W":W})

point_array = []
old_xy = (0,0)

def onMouse(event, x, y, flags, params):
    global point_array,old_xy
    if event == cv2.EVENT_LBUTTONDOWN:
        px = round(x*0.05, 1)
        py = round((H-y)*0.05, 1)
        print({"X":px,"Y":py})
        point_array.append((px,py))
        
        if old_xy != (0,0):
            cv2.line(img, (x, y), old_xy, (255, 0, 0), thickness=3)
        old_xy = (x,y)

cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, onMouse)
cv2.waitKey(0)

cv2.imshow(window_name, img)
cv2.waitKey(0)
print(point_array)




"""
im = cv2.imread('nakano_11f.png')

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

threshold = 100

# 二値化(閾値100を超えた画素を255にする。)
ret, img_thresh = cv2.threshold(im_gray, threshold, 255, cv2.THRESH_BINARY)

height= img_thresh.shape[0]
width = img_thresh.shape[1]

print("height,width: " + str(height)+","+str(width))

grid_map = np.where(img_thresh<1, 0, 1)
"""
