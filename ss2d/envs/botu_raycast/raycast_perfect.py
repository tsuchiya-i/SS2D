#coding:utf-8

import numpy as np
import math

from time import time

class raycast(object):
    def __init__(self, pose, grid_map, grid_height, grid_width, xyreso, yawreso, min_range, max_range, view_angle):
        self.pose      = pose
        self.grid_map  = grid_map
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.xyreso    = xyreso
        self.yawreso   = yawreso
        self.min_range = min_range
        self.max_range = max_range
        self.view_angle = view_angle

    def min2(self,x,y):
        if x < y:
            return x
        else:
            return y
    def negative2zero(self,x):
        if x < 0:
            return 0
        else:
            return x

    def calc_straight_line(self,x1, y1, x2, y2):
        '''
        Returns a list of pixels through which pixel(i1, j1) and pixel(i2, j2) pass
        https://teratail.com/questions/149396
        '''
        points = []
    
        if x1 == x2:
            for i in range(y1,y2+np.sign(y2-y1),np.sign(y2-y1)):
                points.append([x1,i])
            return np.array(points).astype(int)
        if y1 == y2:
            for i in range(x1,x2+np.sign(x2-x1),np.sign(x2-x1)):
                points.append([i,y1])
            return np.array(points).astype(int)
    
        x1, y1 = x1 + 0.5, y1 + 0.5 
        x2, y2 = x2 + 0.5, y2 + 0.5 
        x, y = x1, y1
        cell_w, cell_h = 1, 1  
        step_x = np.sign(x2 - x1) 
        step_y = np.sign(y2 - y1)
        delta_x = cell_w / abs(x2 - x1)
        delta_y = cell_h / abs(y2 - y1)
        max_x = delta_x * (x1 / cell_w % 1)
        max_y = delta_y * (y1 / cell_h % 1)
    
        points.append([x, y])
        reached_x, reached_y = False, False 
        while not (reached_x and reached_y):
            if max_x < max_y:
                max_x += delta_x
                x += step_x
            else:
                max_y += delta_y
                y += step_y
            points.append([x, y]) 
            if (step_x > 0 and x >= x2) or (step_x <= 0 and x <= x2):
                reached_x = True
            if (step_y > 0 and y >= y2) or (step_y <= 0 and y <= y2):
                reached_y = True
    
        return np.array(points).astype(int)

    def raycasting(self):
        lidar_num = int(math.radians(360)/self.yawreso)
        id_list = []
        top_list = []
        raycast_data = []
        pxy_list = []
        total = 0
        for i in range(lidar_num):
            straight_pixel_list = []
            angle = i * self.yawreso
            global_angle = angle + self.pose[2]
            if angle < self.view_angle/2 or angle > math.radians(360)-self.view_angle/2:
                laser_x = self.max_range*math.cos(global_angle)
                laser_y = self.max_range*math.sin(global_angle)
                top_x = self.pose[0]+laser_x
                top_y = self.pose[1]+laser_y
                pose_xp = int(self.pose[0]/self.xyreso)
                pose_yp = int(self.grid_height-(self.pose[1]/self.xyreso))
                top_xp = self.min2(int(top_x/self.xyreso),self.grid_width-1)
                top_yp = self.min2(int(self.grid_height-(top_y/self.xyreso)),self.grid_height-1)
                top_xp = self.negative2zero(top_xp)
                top_yp = self.negative2zero(top_yp)

                #0.0007s
                stime = time()
                straight_pixel_list = self.calc_straight_line(pose_yp,pose_xp,top_yp,top_xp)
                total += time()-stime
                for pix in straight_pixel_list:#0.0003s
                    if self.grid_map[pix[0]][pix[1]] > 0:
                        xy = np.array([pix[1]*self.xyreso,(self.grid_height-pix[0]-1)*self.xyreso])
                        d = np.linalg.norm(xy-self.pose[:2])
                        break
                else:
                    d = self.max_range
                x = d*math.cos(global_angle)
                y = d*math.sin(global_angle)

                raycast_data.append([x, y, angle, d, i, 0])

        raycast_data = np.array(raycast_data)
        #print(total)
        return raycast_data
