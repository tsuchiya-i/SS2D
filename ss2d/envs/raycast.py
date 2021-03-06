#coding:utf-8

import random
import numpy as np
import math

class raycast(object):
    def __init__(self, pose, grid_map, grid_height, grid_width, xyreso, yawreso, min_range, max_range, view_angle, lidar_error):
        self.pose      = pose
        self.grid_map  = grid_map
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.xyreso    = xyreso
        self.yawreso   = yawreso
        self.min_range = min_range
        self.max_range = max_range
        self.view_angle = view_angle
        self.lidar_error = lidar_error

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
    def my_sign(self, number):
        return (number > 0) - (number < 0)

    def calc_straight_line(self, i1, j1, i2, j2):
        '''ピクセル (i1, j1) と ピクセル (i2, j2) が通るピクセルの一覧を返す
        '''
        if i1 == i2 and j1 == j2:
            points_i = np.array([i1])
            points_j = np.array([j1])
        elif i1 == i2:
            points_j = np.arange(j1, j2+self.my_sign(j2-j1), self.my_sign(j2-j1))
            points_i = np.full(len(points_j),i1)
        elif j1 == j2:
            points_i = np.arange(i1, i2+self.my_sign(i2-i1), self.my_sign(i2-i1))
            points_j = np.full(len(points_i),j1)
        else:
            di = i2 - i1
            dj = j2 - j1
            adi = abs(di)
            adj = abs(dj)
    
            max_d = adi if adi>adj else adj
    
            step_i = di/(max_d)
            step_j = dj/(max_d)
            points_i = np.round(np.arange(i1, i2+step_i/2, step_i))
            points_j = np.round(np.arange(j1, j2+step_j/2, step_j))
            if len(points_i) > len(points_j):
                points_i = points_i[1:]
            elif len(points_i) < len(points_j):
                points_j = points_j[1:]
                
        points = np.column_stack([points_i,points_j])
        return np.array(points).astype(int)

    def raycasting(self):
        lidar_num = int(math.radians(360)/self.yawreso)
        id_list = []
        top_list = []
        raycast_data = []
        pxy_list = []
        total = 0
        pose_xp = self.min2(int(self.pose[0]/self.xyreso),self.grid_width-1)
        pose_yp = self.min2(int(self.grid_height-(self.pose[1]/self.xyreso)),self.grid_height-1)
        for i in range(lidar_num):
            straight_pixel_list = []
            angle = i * self.yawreso
            global_angle = angle + self.pose[2]
            if angle <= self.view_angle/2 or angle > math.radians(360)-self.view_angle/2:
                laser_x = self.max_range*math.cos(global_angle)
                laser_y = self.max_range*math.sin(global_angle)
                top_x = self.pose[0]+laser_x
                top_y = self.pose[1]+laser_y
                top_xp = int(top_x/self.xyreso)
                top_yp = int(self.grid_height-(top_y/self.xyreso))

                straight_pixel_list = self.calc_straight_line(pose_yp,pose_xp,top_yp,top_xp)
                for pix in straight_pixel_list:#0.00020~35s
                    if pix[0]<0 or pix[0]>self.grid_height-1 or pix[1]<0 or pix[1]>self.grid_width-1:
                        lidar_top_pix = [top_yp,top_xp]
                        d = self.max_range
                        human_TF = 0
                        break
                    if self.grid_map[pix[0]][pix[1]] > 0:
                        xy = np.array([pix[1]*self.xyreso,(self.grid_height-pix[0]-1)*self.xyreso])
                        d = np.linalg.norm(xy-self.pose[:2])
                        lidar_top_pix = pix
                        if self.grid_map[pix[0]][pix[1]] == 2:
                            human_TF = 1
                        else:
                            human_TF = 0
                        break
                else:
                    d = self.max_range
                    lidar_top_pix = [top_yp,top_xp]
                    human_TF = 0
                
                if self.lidar_error > 0:
                    d = d + random.uniform(-self.lidar_error, self.lidar_error)
                raycast_data.append([angle, d, i, human_TF, lidar_top_pix[0],lidar_top_pix[1]])#human_TF(0,1)

        raycast_data = np.array(raycast_data)
        return raycast_data
