#coding:utf-8

import numpy as np
import math

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

    def calc_obstacle_position(self):
        sxx = int(self.pose[0]/self.xyreso) - int(self.max_range/self.xyreso)
        fxx = int(self.pose[0]/self.xyreso) + int(self.max_range/self.xyreso)
        syy = (self.grid_height-1)-(int(self.pose[1]/self.xyreso)+int(self.max_range/self.xyreso))
        fyy = (self.grid_height-1)-(int(self.pose[1]/self.xyreso)-int(self.max_range/self.xyreso))
        if sxx<0: sxx = 0
        if fxx>self.grid_width-1: fxx = self.grid_width-1
        if syy<0: syy = 0
        if fyy>self.grid_height-1: fyy = self.grid_height-1
        
        search_map = self.grid_map[syy:fyy,sxx:fxx]
        obstacle_grid = np.where(search_map>0)

        pixel2xy = np.array([(obstacle_grid[1]+sxx)*self.xyreso, (self.grid_height-(obstacle_grid[0]+syy))*self.xyreso])
        obstacle_position =  np.column_stack((pixel2xy[0],pixel2xy[1]))

        obstacle_list = search_map[search_map>0]
        self.human_list = np.where(obstacle_list==2,1,0)
        return obstacle_position

    def transform(self, x, y, obstacle_position):
        transform_position = obstacle_position-[x,y]
        return transform_position

    def rotation(self, radian, obstacle_position):
        rotation = np.array([[math.cos(radian),   -math.sin(radian)],
                             [math.sin(radian),  math.cos(radian)]]);
        rotation_position = np.dot(obstacle_position, rotation)
        return rotation_position

    def raycasting(self):
        obstacle_position  = self.calc_obstacle_position()
        transform_position = self.transform(self.pose[0], self.pose[1], obstacle_position)
        rotation_position = self.rotation(self.pose[2], transform_position)
        
        lidar_num = int(math.radians(360)/self.yawreso)
        id_list = []
        for i in range(lidar_num):
            angle = i * self.yawreso
            if angle < self.view_angle/2 or angle > math.radians(360)-self.view_angle/2:
                id_list.append(i)

        
        angle_list = np.arctan2(rotation_position.T[1],rotation_position.T[0])
        angle_fid_list = angle_list/self.yawreso
        angle_fid_list = np.where(angle_fid_list<0,angle_fid_list+math.radians(360)//self.yawreso,angle_fid_list)
        angle_id_list = np.round(angle_fid_list)

        dist_list = np.linalg.norm(rotation_position,axis=1)
        dist_list = np.where(dist_list>self.max_range,self.max_range,dist_list)
        dist_list = np.where(dist_list<self.min_range,self.min_range,dist_list)

        raycast_data = []
        ex_count = 0
        for id in id_list:
            angle_dist_list = dist_list[angle_id_list==id]
            angle_human_list = self.human_list[angle_id_list==id]
            if not len(angle_dist_list):
                angle_dist_list = np.array([self.max_range])
                angle_human_list = np.array([0])
            min_index = np.argmin(angle_dist_list)
            a = id * self.yawreso
            if ex_count > 0:
                d = self.min2(angle_dist_list[min_index],next_d)
                ex_count -= 1
            else:
                d = angle_dist_list[min_index]
            x = d*math.cos(a)
            y = d*math.sin(a)
            humanTF = angle_human_list[min_index]
            raycast_data.append([x, y, a, d, id, humanTF])#humanTF:0or1
            """
            if np.arctan2(self.xyreso,d)>self.yawreso:
                ex_count = int(np.arctan2(self.xyreso,d)/self.yawreso)
                next_d = d
            """
        raycast_data = np.array(raycast_data)
        
        return raycast_data

