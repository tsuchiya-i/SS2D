#coding:utf-8
import numpy as np
import math
import random
import pickle
import os

import cv2
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import rvo2

from ss2d.envs.raycast import *

target_color = True


class configClass():
    def __init__(self):
        self.thresh_map = None #image array(2D)
        self.color_map = None
        self.start_points = [] #[[x1,y1],[x2,y2],,] 
        self.goal_points = []
        self.human_points = []
        self.reso = 0.05 #m/pix
        self.world_dt = 0.1 #sec
        self.robot_r = 0.3 #m
        self.lidar_max = 10 #m
        self.lidar_min = 0.2 #m
        self.lidar_angle = 90 #deg
        self.lidar_reso = 10 #deg
        self.human_n = 0 #person
        self.human_detect = True #bool
        self.console_output = True #bool


class SS2D_env(gym.Env):
    def __init__(self):
        if os.path.exists(__file__.replace("environment.py","")+"test_config.bin"):
            with open(__file__.replace("environment.py","")+"test_config.bin", mode='rb') as f:
                self.config = pickle.load(f)
                print("----test config data----")
            os.remove(__file__.replace("environment.py","")+"test_config.bin")
        else:
            with open(__file__.replace("environment.py","")+"config.bin", mode='rb') as f:
                self.config = pickle.load(f)
                print("----Load config data----")

        self.show = self.config.console_output #bool
        # world param
        self.dt = self.config.world_dt #[s]
        self.map_height= 0 #[pix]
        self.map_width = 0 #[pix]
        self.max_dist = 1000
        self.world_time= 0.0 #[s]
        self.step_count = 0 #[]
        self.reset_count = 0 #[]

        # robot param
        self.robot_radius = self.config.robot_r #[m]
        # state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)
        self.state = np.array([0.0, 0.0, math.radians(0), 0.0, 0.0])

        # action param
        self.max_velocity = 0.8   # [m/s]
        self.min_velocity = -0.4  # [m/s]
        self.min_angular_velocity = math.radians(-40)  # [rad/s]
        self.max_angular_velocity = math.radians(40) # [rad/s]

        # human param
        self.human_n = self.config.human_n
        self.human_vel_min = 0.8
        self.human_vel_max = 0.8
        self.human_radius = 0.25 #[m]
        self.nearest_j = [0] * self.human_n #

        # lidar param
        self.yawreso = math.radians(self.config.lidar_reso) # ※360から割り切れる(1~360)[rad]
        self.min_range = self.config.lidar_min # [m]
        self.max_range = self.config.lidar_max # [m]
        self.view_angle = math.radians(self.config.lidar_angle) #[rad]
        self.lidarnum = int(int(self.view_angle/(2*self.yawreso))*2+1)

        #observe param
        self.human_detect = self.config.human_detect #F:(lidar+goal) T:(lidar+human+goal)

        # set action_space (velocity[m/s], omega[rad/s])
        self.action_low  = np.array([self.min_velocity, self.min_angular_velocity]) 
        self.action_high = np.array([self.max_velocity, self.max_angular_velocity]) 
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)
        # set observation_space
        if self.human_detect == 0:
            self.observation_low = np.concatenate([[0.0]*self.lidarnum ,[0.0, -math.pi,]],0)
            self.observation_high = np.concatenate([[self.max_range]*self.lidarnum ,[self.max_dist,-math.pi]],0)
        elif self.human_detect == 1:
            self.observation_low = np.concatenate([[0.0]*(self.lidarnum*2) ,[0.0, -math.pi,]],0)
            self.observation_high = np.concatenate([[self.max_range]*(self.lidarnum*2) ,[self.max_dist,-math.pi]],0)
        self.observation_space = spaces.Box(low = self.observation_low, high = self.observation_high, dtype=np.float32)

        #way point
        #self.way_point_set() #default:0
        self.waypoints = np.array(self.config.start_points)
        self.human_waypoints = np.array(self.config.human_points)
        self.goal_points = np.array(self.config.goal_points)
        self.near_n = 2 #Destination selection(current location+near_n)
        self.neighbors_vector_set(self.near_n,neighbors_id=0)

        #rendering
        self.vis_lidar = True
        self.viewer = None

    def reset(self):
        self.set_image_map()
        self.robot_r_cell = int(self.robot_radius/self.xyreso) #[cell]
        self.human_r_pix = int(self.human_radius/self.xyreso)
        
        self.start_p_num = random.randint(0, len(self.waypoints)-1)
        goal_p_num = random.randint(0, len(self.goal_points)-1)
        if (self.goal_points == self.waypoints).all():
            while self.start_p_num==goal_p_num:
                goal_p_num= random.randint(0, len(self.waypoints)-1)
        self.goal = self.goal_points[goal_p_num]
        self.start = self.waypoints[self.start_p_num]

        self.state = np.array([self.waypoints[self.start_p_num][0], self.waypoints[self.start_p_num][1], \
            math.radians(random.uniform(179, -179)),0,0])

        #initial human pose
        self.init_human()
        self.rvo_robot = self.sim.addAgent((self.state[0],self.state[1]))
        self.sim.setAgentVelocity(self.rvo_robot, (0.0,0.0))
        #rvo map set
        self.rvomap_set()
        #reset goal_d and goal_a 
        self.distgoal = self.calc_goal_info()
        self.old_distgoal = self.distgoal
        #reset observe
        self.observation = self.observe()
        self.done = False
        # world_time reset
        self.world_time = 0.0
        self.reset_count += 1
        return self.observation

    def step(self, action):
        self.step_count += 1
        self.world_time += self.dt
        
        self.human_step()
        self.robot_step(action)
        self.sim.setAgentPosition(self.rvo_robot,(self.state[0],self.state[1]))
        self.observation = self.observe()
        reward = self.reward(action)
        self.done = self.is_done(self.show)
        self.old_distgoal = self.distgoal

        return self.observation, reward, self.done, {}
    
    def observe(self):
        # Raycasting
        Raycast = raycast(self.state[0:3], self.map, self.map_height,self.map_width, 
                                self.xyreso, self.yawreso,
                                self.min_range, self.max_range,self.view_angle)
        self.lidar = Raycast.raycasting()

        human_dist_data = self.lidar[:, 3]*self.lidar[:, 1]
        human_dist_data = np.where(human_dist_data==0,10,human_dist_data)

        if self.human_detect == 0:
            observation = self.lidar[:, 1]
        elif self.human_detect == 1:
            observation = np.concatenate([self.lidar[:, 1], human_dist_data], 0)

        self.distgoal = self.calc_goal_info()
        distgoal_norm = np.array([self.distgoal[0],self.distgoal[1]])
        observation = np.concatenate([observation, distgoal_norm], 0)

        return observation#lidar,human,dist,angle_dist(clock wise)

    def reward(self, action):
        if self.is_goal():
            rwd = 25 
        elif self.is_collision(False):
            if self.collision_factor == 1:
                rwd = -25
            elif self.collision_factor == 2:
                rwd = -40
        elif not self.is_movable():
            rwd = -25
        else:
            if self.lidar[np.argmin(self.lidar[:, 1]), 1] < self.robot_radius*2:
                wall_rwd = -1.0
            else:
                wall_rwd = 0.0
            vel_rwd = (action[0]-self.max_velocity)/self.max_velocity
            dist_rwd = (self.old_distgoal[0]-self.distgoal[0])/(self.max_velocity*self.dt)
            angle_rwd = (abs(self.old_distgoal[1])-abs(self.distgoal[1]))/(self.max_angular_velocity*self.dt)
            time_reward = -self.world_time/(1500*self.dt) #max 1500steps
            rwd = (vel_rwd + 2*dist_rwd + 2*angle_rwd)/5 + wall_rwd + time_reward
        return rwd

    def set_image_map(self, scale=1):
        #self.map_height, width, self.map, self.original_map self.max_dist set
        img_thresh = self.config.thresh_map
        self.viewer = None
        if img_thresh.shape[0] != self.map_height or img_thresh.shape[1] != self.map_width:
            self.map_height= img_thresh.shape[0] #[pix]
            self.map_width = img_thresh.shape[1] #[pix]
            if self.viewer != None:
                self.viewer.close()
                self.viewer = None
        self.map = np.where(img_thresh>100, 0, 1)
        self.original_map = np.where(img_thresh>100, 0, 1)
        self.xyreso = self.config.reso
        self.max_dist = (self.map_height+self.map_width)*self.xyreso

    def is_goal(self, show=False):
        if math.sqrt( (self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2 ) <= self.robot_radius*3:
            if show:
                print("Goal")
            return True
        else:
            return False

    def is_movable(self, show=False):
        x = int(self.state[0]/self.xyreso)
        y = int(self.state[1]/self.xyreso)
        if(0<=x<self.map_width and 0<=y<self.map_height and self.map[self.map_height-1-y,x] == 0):
            return True
        else:
            if show:
                print("(%f, %f) is not movable area" % (x*self.xyreso, y*self.xyreso))
            return False

    def is_collision(self, show=False):
        x = int(self.state[0]/self.xyreso) #[cell]
        y = int(self.state[1]/self.xyreso) #[cell]
        sx = self.max2(x-self.robot_r_cell,0)
        fx = self.min2(x+self.robot_r_cell,self.map_width-1)
        sy = self.max2((self.map_height-1)-(y+self.robot_r_cell),0)
        fy = self.min2((self.map_height-1)-(y-self.robot_r_cell),self.map_height-1)

        obstacle = np.where(0<self.map[sy:fy,sx:fx])
        h_obstacle = np.where(2==self.map[sy:fy,sx:fx])
        if len(obstacle[0]):
            if len(h_obstacle[0]):
                self.collision_factor = 2
                if show:
                    print("(%f, %f) of human collision" % (x*self.xyreso, y*self.xyreso))
            else:
                self.collision_factor = 1
                if show:
                    print("(%f, %f) of collision" % (x*self.xyreso, y*self.xyreso))
            return True
        self.collision_factor = 0
        return False

    def is_done(self, show=False):
        if self.is_collision(show):
            return True
        elif (not self.is_movable(show)):
            return True
        elif self.is_goal(show):
            return True
        else:
            return False

    def calc_goal_info(self):
        d = np.linalg.norm(self.goal-self.state[:2])
        t_theta = np.angle(complex(self.goal[0]-self.state[0], self.goal[1]-self.state[1]))
        if t_theta < 0:
            t_theta = math.pi + (math.pi+t_theta)
        theta = t_theta
        if self.state[2] < math.pi:
            if t_theta > self.state[2]+math.pi:
                theta=t_theta-2*math.pi
        if self.state[2] > math.pi:
            if 0 < t_theta < self.state[2]-math.pi:
                theta=t_theta+2*math.pi
        anglegoal = theta-self.state[2]
        return np.array([d, anglegoal])

    def init_human(self):
        self.sim = rvo2.PyRVOSimulator(1/10., 1.5, 5, 1.5, 2, self.human_radius, 2)
        self.human_state = []
        self.human_vel = []
        rand_num_his = []
        #human pix on map
        self.target_position = [[-1,-1]] * self.human_n
        
        #spawn on waypoints
        if (self.human_waypoints == self.waypoints).all():
            while len(rand_num_his) < self.human_n:
                rand_num = random.randint(0, len(self.waypoints)-1)
                if (not rand_num in rand_num_his) and (rand_num != self.start_p_num):
                    rand_num_his.append(rand_num)
                    self.hstart_p = (self.waypoints[rand_num])
                    self.human_state.append(self.sim.addAgent((self.hstart_p[0], self.hstart_p[1])))
                    human_vel = random.uniform(self.human_vel_min, self.human_vel_max)
                    self.human_vel.append(human_vel)
        else:
            while len(rand_num_his) < self.human_n:
                rand_num = random.randint(0, len(self.human_waypoints)-1)
                if not rand_num in rand_num_his:
                    rand_num_his.append(rand_num)
                    self.hstart_p = (self.human_waypoints[rand_num])
                    self.human_state.append(self.sim.addAgent((self.hstart_p[0], self.hstart_p[1])))
                    human_vel = random.uniform(self.human_vel_min, self.human_vel_max)
                    self.human_vel.append(human_vel)


    def way_point_set(self,waypoints_id=0):
        if waypoints_id == 0:
            self.waypoints = np.array([[9, 8.5], [9, 12], [9, 16], [9, 19], [15, 19], [20, 19], [25, 19], [30, 19], [35, 19], [40.5, 18], [40.5, 16], [40.5, 13], [40.5, 11], [40.5, 8.5], [35, 8.5], [30, 8.5], [25, 8.5], [20, 8.5], [15, 8.5], [48, 11], [57, 11], [63, 11], [48, 16], [57, 16], [63, 16], [48, 13], [3, 14]])
            self.human_waypoints = np.array([[8.8, 19.0], [40.4, 18.2], [40.7, 16.0], [47.7, 16.1], [63.2, 16.0], [63.2, 10.8], [47.6, 10.8], [40.9, 10.6], [40.5, 8.], [8.8, 8.1]])


    def neighbors_vector_set(self,n=2,neighbors_id=0): #n near points
        if neighbors_id == 0:
            self.neighbors_array = []
            for point in self.human_waypoints:
                dist_array = np.linalg.norm(point-self.human_waypoints,axis=1)
                sort_index = np.argsort(dist_array)
                self.neighbors_array.append(sort_index[:n+1])
        elif neighbors_id == 1:
            # same points and near points
            self.neighbors_array = np.array([[0,1,9],[1,2,0],[2,3,1,7],[3,4,2,6],[4,5,3],[5,6,4],[6,7,5,3],[7,8,6,2],[8,9,7],[9,0,8]])

    def rvomap_set(self,rvo_map_id=0):
        if rvo_map_id == 0:
            pass
            """
            for points in self.config.rvomap:
                map_line = points
                obstacle = self.sim.addObstacle(map_line)
            map_line = []#self.config.rvomap
            """
        if rvo_map_id == 1:
            map_line = [(7.6, 22.9), (10.3, 22.7), (10.2, 19.9), (41.7, 19.5), (41.7, 16.9), (53.4, 16.9), (53.4, 19.5), (58.9, 19.5), (59.0, 16.9), (63.8, 17.1), (63.9, 10.0), (59.2, 10.0), (59.2, 7.5), (53.9, 7.4), (53.8, 9.8), (53.0, 9.8), (53.2, 1.5), (47.1, 1.2), (46.9, 9.6), (51.6, 9.7), (41.6, 9.7), (41.5, 7.2), (7.5, 7.1), (7.5, 12.0), (0.7, 12.0), (0.8, 15.2), (7.5, 15.1), (7.5, 18.2), (3.6, 18.2), (3.5, 20.1), (4.4, 20.1), (4.5, 19.2), (7.6, 19.2)]
            map_line2 = [(10.2, 16.6), (10.2, 17.9), (12.8, 17.8), (12.8, 17.2), (15.8, 17.2), (15.8, 17.7), (16.5, 17.8), (16.9, 17.2), (20.0, 17.2), (20.0, 17.6), (25.2, 17.7), (25.4, 17.0), (28.8, 16.9), (28.6, 17.4), (30.0, 17.4), (30.1, 16.9), (35.2, 16.9), (35.2, 17.5), (36.5, 17.2), (36.5, 16.9), (39.5, 16.8), (39.7, 9.7), (36.3, 9.7), (36.3, 9.2), (35.1, 9.2), (35.1, 9.7), (30.2, 9.6), (30.1, 9.1), (28.5, 9.2), (28.4, 9.8), (25.4, 9.7), (25.4, 9.1), (20.0, 9.1), (20.0, 9.6), (16.7, 9.6), (16.7, 9.2), (15.8, 9.2), (15.7, 9.6), (12.5, 9.8), (12.5, 9.2), (10.2, 9.1), (10.1, 11.1), (11.6, 11.1), (11.7, 16.5)]
            map_line2.reverse()
            map_line3 = [(41.7, 15.0), (46.7, 14.9), (45.9, 14.7), (45.7, 11.9), (46.6, 11.8), (41.7, 11.4)]
            map_line3.reverse()
            map_line4 = [(48.5, 15.1), (62.6, 15.3), (62.5, 11.8), (48.6, 11.6), (49.4, 11.9), (49.3, 14.8)]
            map_line4.reverse()
            o1 = self.sim.addObstacle(map_line)
            o2 = self.sim.addObstacle(map_line2)
            o3 = self.sim.addObstacle(map_line3)
            o4 = self.sim.addObstacle(map_line4)
        self.sim.processObstacles()

    def robot_step(self, action):
        for i in range(len(action)):
            if action[i] < self.action_low[i]:
                action[i] = self.action_low[i]
            if action[i] > self.action_high[i]:
                action[i] = self.action_high[i]

        self.state[0] += action[0] * math.cos(self.state[2]) * self.dt
        self.state[1] += action[0] * math.sin(self.state[2]) * self.dt
        self.state[2] += action[1] * self.dt
        if self.state[2]<0.0:
            self.state[2] += math.pi * 2.0
        elif math.pi * 2.0 < self.state[2]:
            self.state[2] -= math.pi * 2.0
        self.state[3] = action[0]
        self.state[4] = action[1]

    def human_step(self):
        self.map = self.original_map.copy()
        total = 0
        for i in range(len(self.human_state)):
            velvec = self.set_rvo_velocity(i,self.human_vel[i])
            self.sim.setAgentPrefVelocity(self.human_state[i], velvec)
            human_pix_i = (self.map_height-1)-int(self.sim.getAgentPosition(self.human_state[i])[1]/self.xyreso)
            human_pix_j = int(self.sim.getAgentPosition(self.human_state[i])[0]/self.xyreso)
            human_pix_si = self.max2(human_pix_i-self.human_r_pix,0)
            human_pix_sj = self.max2(human_pix_j-self.human_r_pix,0)
            human_pix_fi = self.min2(human_pix_i+self.human_r_pix,self.map_height-1)+1
            human_pix_fj = self.min2(human_pix_j+self.human_r_pix,self.map_width-1)+1
            self.map[human_pix_si:human_pix_fi,human_pix_sj:human_pix_fj] = 2
        self.sim.doStep()

    def set_rvo_velocity(self, i, human_vel):
        now_position = np.array(self.sim.getAgentPosition(self.human_state[i]))
        if self.target_position[i][0] == -1:
            for j in range(len(self.human_waypoints)):#search all points
                point_dist = np.linalg.norm(self.human_waypoints[j]-now_position)
                if j==0 or nearest_dist > point_dist:
                    nearest_dist = point_dist
                    min_j = j
            self.target_position[i] = self.human_waypoints[min_j]
        if np.linalg.norm(self.target_position[i] - now_position) < self.human_radius:
            for j in range(len(self.human_waypoints)):#search all points
                point_dist = np.linalg.norm(self.human_waypoints[j]-now_position)
                if j==0 or nearest_dist > point_dist:
                    nearest_dist = point_dist
                    min_j = j
            self.nearest_j[i] = min_j
            self.target_point_num = [0] * self.human_n
            self.target_point_num[i] = random.randint(0, len(self.neighbors_array[min_j])-1)
            target_point = self.neighbors_array[self.nearest_j[i]][self.target_point_num[i]]
            target_point = self.human_waypoints[int(target_point)]
            self.target_position[i] = target_point
        target_vector = self.target_position[i] - now_position
        target_vel = target_vector/np.linalg.norm(target_vector) * human_vel

        return (target_vel[0], target_vel[1])
    
    def render(self, mode='human', close=False):
        if self.viewer is None:
            max_s = max(self.config.color_map.shape[0],
                    self.config.color_map.shape[1])
            self.mag = 800/max_s
            self.d_robot_r = int(self.robot_radius/self.xyreso*self.mag)
            self.d_human_r = int(self.human_radius/self.xyreso*self.mag)
            self.viewer = cv2.resize(self.config.color_map,
                    dsize=None,fx=self.mag,fy=self.mag)
            cv2.circle(self.viewer,(int(self.start[0]/self.xyreso*self.mag),
                int(((self.map_height-1)-self.start[1]/self.xyreso)*self.mag)),
                self.d_robot_r, (235,206,135), thickness=-1)
            cv2.circle(self.viewer,(int(self.goal[0]/self.xyreso*self.mag),
                int(((self.map_height-1)-self.goal[1]/self.xyreso)*self.mag)),
                self.d_robot_r, (0,0,255), thickness=-1)

        x = int(self.state[0]/self.xyreso*self.mag) 
        y = int(((self.map_height-1)-self.state[1]/self.xyreso)*self.mag)

        disply = self.viewer.copy()
        robot = cv2.circle(disply, (x, y), self.d_robot_r, (200,0,0), thickness=-1)
        for lidar in self.lidar:
            if lidar[3]==1:
                color = (0,0,255)
            elif lidar[0]==0:
                color = (170,205,102)
            else:
                color = (208,224,64)
            scan = cv2.line(disply, (x, y),(int(lidar[4][1]*self.mag),int(lidar[4][0]*self.mag)), color, thickness=2)
        for sim_id in self.human_state:
            hx = int(self.sim.getAgentPosition(sim_id)[0]/self.xyreso*self.mag)
            hy = int(((self.map_height-1)-self.sim.getAgentPosition(sim_id)[1]/self.xyreso)*self.mag)
            human = cv2.circle(disply, (hx, hy), self.human_r_pix, (226,43,138), thickness=-1)

        cv2.imshow('SS2D',disply)
        cv2.waitKey(1)

    def close(self):
        if self.viewer is not None:
            #self.viewer.close()
            self.viewer = None
            cv2.destroyAllWindows()

    def max2(self,a,b):
        if a > b:
            return a
        else:
            return b

    def min2(self,a,b):
        if a < b:
            return a
        else:
            return b
    def my_clip(self,a,min_,max_):
        b = max2(a,min_)
        b = min2(b,max_)
        return b
