#coding:utf-8

import numpy as np
import math
import cv2
import time
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from ss2d.envs.raycast import *

import rvo2

from time import time

wall_switch = False
human_mode = 6 #0:stop 1:straight 2:random 3:bound 4:onedirection 5:RVO_straight 6:RVO_near_waypoint

human_detect = True#True#True
world_map = 1 #0:free 1:nakano11F

mode = 1 #0:normal, 1:simple, 2:test


target_color = True

class SS2D_env(gym.Env):
    def __init__(self):
        self.show = True
        # world param
        self.map_height= 0 #[pix]
        self.map_width = 0 #[pix]
        self.max_dist = 1000
        self.dt = 0.1 #[s]
        self.world_time= 0.0 #[s]
        self.step_count = 0 #[]
        self.reset_count = 0 #[]

        # robot param
        self.robot_radius = 0.3 #[m]
        # state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)
        self.state = np.array([0.0, 0.0, math.radians(0), 0.0, 0.0])

        # action param
        self.max_velocity = 0.8   # [m/s]
        self.min_velocity = -0.4  # [m/s]
        self.min_angular_velocity = math.radians(-40)  # [rad/s]
        self.max_angular_velocity = math.radians(40) # [rad/s]

        # human param
        self.human_n = 20
        self.human_vel_min = 0.8
        self.human_vel_max = 0.8
        self.human_radius = 0.35 #[m]
        self.nearest_j = [0] * self.human_n #

        # lidar param
        self.yawreso = math.radians(10) # ※360から割り切れる(1~360)[rad]
        self.min_range = 0.20 # [m]
        self.max_range = 10.0 # [m]
        self.view_angle = math.radians(90) #[rad]
        self.lidarnum = int(int(self.view_angle/(2*self.yawreso))*2+1)

        #observe param
        self.observe_mode = 1#0:(lidar+goal) 1:(lidar+human+goal)

        # set action_space (velocity[m/s], omega[rad/s])
        self.action_low  = np.array([self.min_velocity, self.min_angular_velocity]) 
        self.action_high = np.array([self.max_velocity, self.max_angular_velocity]) 
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)
        # set observation_space
        if self.observe_mode == 0:
            self.observation_low = np.concatenate([[0.0]*self.lidarnum ,[0.0, -math.pi,]],0)
            self.observation_high = np.concatenate([[self.max_range]*self.lidarnum ,[self.max_dist,-math.pi]],0)
        elif self.observe_mode == 1:
            self.observation_low = np.concatenate([[0.0]*(self.lidarnum*2) ,[0.0, -math.pi,]],0)
            self.observation_high = np.concatenate([[self.max_range]*(self.lidarnum*2) ,[self.max_dist,-math.pi]],0)
        self.observation_space = spaces.Box(low = self.observation_low, high = self.observation_high, dtype=np.float32)

        #way point
        self.way_point_set() #default:0
        self.near_n = 2 #人の行き先の選択肢の数(現在地(停止)＋near_n)
        self.neighbors_vector_set(self.near_n,neighbors_id=1)

        #rendering
        self.vis_lidar = True
        self.viewer = None

    
    # 状態を初期化し、初期の観測値を返す
    def reset(self):
        """
        if self.reset_count == 0:
            self.xyreso = 0.05*4 #[m/pix]
            self.set_image_map(__file__[:-24]+'maps/nakano_11f_sim.png', self.xyreso, 1.0/4)
        if self.reset_count > 0:
            self.xyreso = 0.05*4 #[m/pix]
            self.set_image_map(__file__[:-24]+'maps/paint_map/line_025.png', self.xyreso, 1)
        """
        self.xyreso = 0.05*4 #[m/pix]
        self.set_image_map(__file__[:-24]+'maps/nakano_11f_line025.png',self.xyreso)
        
        self.start_p_num = random.randint(0, len(self.waypoints)-1)
        self.start_p_num = len(self.waypoints)-1
        goal_p_num = random.randint(0, len(self.waypoints)-1)
        while self.start_p_num==goal_p_num:
            goal_p_num= random.randint(0, len(self.waypoints)-1)
        self.start_p = self.waypoints[self.start_p_num]
        self.goal = self.waypoints[goal_p_num]

        #self.state = np.array([self.waypoints[self.start_p_num][0], self.waypoints[self.start_p_num][1], \
                #math.radians(random.uniform(179, -179)),0,0])
        self.state = np.array([self.waypoints[self.start_p_num][0], self.waypoints[self.start_p_num][1], math.radians(0),0,0])

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

    # actionを実行し、結果を返す
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
    
    # 観測結果を表示
    def observe(self):
        # Raycasting
        stime = time()
        Raycast = raycast(self.state[0:3], self.map, self.map_height,self.map_width, 
                                self.xyreso, self.yawreso,
                                self.min_range, self.max_range,self.view_angle)
        self.lidar = Raycast.raycasting()

        human_dist_data = self.lidar[:, 3]*self.lidar[:, 1]
        human_dist_data = np.where(human_dist_data==0,10,human_dist_data)
        #human_detect_data = [self.max_range]*9 #human no detect

        if self.observe_mode == 0:
            observation = self.lidar[:, 1]
        elif self.observe_mode == 1:
            observation = np.concatenate([self.lidar[:, 1], human_dist_data], 0)

        self.distgoal = self.calc_goal_info()
        distgoal_norm = np.array([self.distgoal[0],self.distgoal[1]])
        observation = np.concatenate([observation, distgoal_norm], 0)

        return observation#lidar,human,dist,angle_dist(clock wise)

    # 報酬値を返す
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
            rwd = (vel_rwd + 2*dist_rwd + 2*angle_rwd)/5 + wall_rwd + time_reward#うまく行った
        return rwd

    def set_image_map(self, map_filename, xyreso, scale=1):
        #self.map_height, width, self.map, self.original_map self.max_dist set
        im = cv2.imread(map_filename)
        
        if scale != 1:
            orgHeight, orgWidth = im.shape[:2]
            size = (int(orgWidth*scale), int(orgHeight*scale))
            im = cv2.resize(im, size)
        
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        threshold = 25
        # 二値化(閾値100を超えた画素を255白にする。)
        ret, img_thresh = cv2.threshold(im_gray, threshold, 255, cv2.THRESH_BINARY)

        #cv2.imshow('image',im)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        if img_thresh.shape[0] != self.map_height or img_thresh.shape[1] != self.map_width:
            self.map_height= img_thresh.shape[0] #[pix]
            self.map_width = img_thresh.shape[1] #[pix]
            if self.viewer != None:
                self.viewer.close()
                self.viewer = None

        self.map = np.where(img_thresh>100, 0, 1)
        self.original_map = np.where(img_thresh>100, 0, 1)

        self.xyreso = xyreso
        self.max_dist = (self.map_height+self.map_width)*xyreso

    # ゴールに到達したかを判定
    def is_goal(self, show=False):
        if math.sqrt( (self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2 ) <= self.robot_radius*3:
            if show:
                print("Goal")
            return True
        else:
            return False

    # 移動可能範囲内に存在するか
    def is_movable(self, show=False):
        x = int(self.state[0]/self.xyreso)
        y = int(self.state[1]/self.xyreso)
        if(0<=x<self.map_width and 0<=y<self.map_height and self.map[self.map_height-1-y,x] == 0):
            return True
        else:
            if show:
                print("(%f, %f) is not movable area" % (x*self.xyreso, y*self.xyreso))
            return False

    # 高速衝突判定
    def is_collision(self, show=False):
        x = int(self.state[0]/self.xyreso) #[cell]
        y = int(self.state[1]/self.xyreso) #[cell]
        robot_radius_cell = int(self.robot_radius/self.xyreso) #[cell]
        sx = self.max2(x-robot_radius_cell,0)
        fx = self.min2(x+robot_radius_cell,self.map_width-1)
        sy = self.max2((self.map_height-1)-(y+robot_radius_cell),0)
        fy = self.min2((self.map_height-1)-(y-robot_radius_cell),self.map_height-1)

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

    # 終端状態か確認
    def is_done(self, show=False):
        if self.is_collision(show):
            return True
        elif (not self.is_movable(show)):# or self.is_collision(show) or self.is_goal(show):
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
        #マップ上の人の位置ピクセル格納用
        self.target_position = [[-1,-1]] * self.human_n
        
        #waypoint上にランダムにnum人の人をスポーン
        while len(rand_num_his) < self.human_n:
            rand_num = random.randint(0, len(self.waypoints)-1)
            if (not rand_num in rand_num_his) and (rand_num != self.start_p_num):
                rand_num_his.append(rand_num)
                self.hstart_p = (self.waypoints[rand_num])
                self.human_state.append(self.sim.addAgent((self.hstart_p[0], self.hstart_p[1])))
                human_vel = random.uniform(self.human_vel_min, self.human_vel_max)
                self.human_vel.append(human_vel)

    def way_point_set(self,waypoints_id=0):
        if waypoints_id == 0:
            self.waypoints = np.array([[9, 8.5], [9, 12], [9, 16], [9, 19], [15, 19], [20, 19], [25, 19], [30, 19], [35, 19], [40.5, 18], [40.5, 16], [40.5, 13], [40.5, 11], [40.5, 8.5], [35, 8.5], [30, 8.5], [25, 8.5], [20, 8.5], [15, 8.5], [48, 11], [57, 11], [63, 11], [48, 16], [57, 16], [63, 16], [48, 13], [3, 14]])
            self.human_waypoints = np.array([[8.8, 19.0], [40.4, 18.2], [40.7, 16.0], [47.7, 16.1], [63.2, 16.0], [63.2, 10.8], [47.6, 10.8], [40.9, 10.6], [40.5, 8.], [8.8, 8.1]])


    def neighbors_vector_set(self,n=2,neighbors_id=0): #n個の近傍ウェイポイントセット
        if neighbors_id == 0:
            self.neighbors_array = []
            for point in self.human_waypoints:
                dist_array = np.linalg.norm(point-self.human_waypoints,axis=1)
                sort_index = np.argsort(dist_array)
                self.neighbors_array.append(sort_index[:n+1])
        elif neighbors_id == 1:
            #１つめはその場のポイント番号
            self.neighbors_array = np.array([[0,1,9],[1,2,0],[2,3,1,7],[3,4,2,6],[4,5,3],[5,6,4],[6,7,5,3],[7,8,6,2],[8,9,7],[9,0,8]])

    def rvomap_set(self,rvo_map_id=0):
        #if world_map == 0:
        #    map_line = [(1.8, 48.1), (69.7, 48.1), (69.5, 1.7), (1.9, 1.7)]
        #    o1 = self.sim.addObstacle(map_line)
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
            stime = time()
            velvec = self.set_rvo_velocity(i,self.human_vel[i])
            total += time()-stime 
            self.sim.setAgentPrefVelocity(self.human_state[i], velvec)

            human_pix_i = (self.map_height-1)-int(self.sim.getAgentPosition(self.human_state[i])[1]/self.xyreso)
            human_pix_j = int(self.sim.getAgentPosition(self.human_state[i])[0]/self.xyreso)
            
            human_r_pix = int(self.human_radius/self.xyreso)
            human_pix_si = self.max2(human_pix_i-human_r_pix,0)
            human_pix_sj = self.max2(human_pix_j-human_r_pix,0)
            human_pix_fi = self.min2(human_pix_i+human_r_pix,self.map_height-1)+1
            human_pix_fj = self.min2(human_pix_j+human_r_pix,self.map_width-1)+1

            self.map[human_pix_si:human_pix_fi,human_pix_sj:human_pix_fj] = 2
        print("##set_rvo_velocity total time##")
        print(total)
        self.sim.doStep()

    def set_rvo_velocity(self, i, human_vel):#iは人ナンバー
        #stime = time()
        now_position = np.array(self.sim.getAgentPosition(self.human_state[i]))
        #最近傍ウェイポイント
        if self.target_position[i][0] == -1:
            for j in range(len(self.human_waypoints)):#ウェイポイント全探索
                point_dist = np.linalg.norm(self.human_waypoints[j]-now_position)
                if j==0 or nearest_dist > point_dist:
                    nearest_dist = point_dist
                    min_j = j
            self.target_position[i] = self.human_waypoints[min_j]
        if np.linalg.norm(self.target_position[i] - now_position) < self.human_radius:
            for j in range(len(self.human_waypoints)):#ウェイポイント全探索
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

        #print((time()-stime)*30)
        return (target_vel[0], target_vel[1])


    # レンダリング
    def render(self, mode='human', close=False):
        screen_width  = self.map_width
        screen_height = self.map_height
        scale_width = screen_width / float(self.map_width) 
        scale_height = screen_height / float(self.map_height)

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # wall
            if wall_switch: 
                for i in range(screen_height):
                    for j in range(screen_width):
    
                        if self.original_map[i][j] == 1:
                            wall = rendering.make_capsule(1, 1)
                            self.walltrans = rendering.Transform()
                            wall.add_attr(self.walltrans)
                            wall.set_color(0.2, 0.4, 1.0)
                            self.walltrans.set_rotation(0)
                            self.viewer.add_geom(wall)
                            self.walltrans.set_translation(j, screen_height-i)
                            self.walltrans.set_rotation(0)
                            self.viewer.add_geom(wall)

            # huaman_waypoints
            """
            for point in self.human_waypoints:
                waypoint = rendering.make_circle(self.robot_radius/self.xyreso*scale_width)
                self.waypointtrans = rendering.Transform()
                waypoint.add_attr(self.waypointtrans)
                waypoint.set_color(1.0, 0.5, 0.0)
                self.waypointtrans.set_translation(point[0]/self.xyreso*scale_width, 
                        point[1]/self.xyreso*scale_height)
                self.viewer.add_geom(waypoint)
            """
            # waypoints
            for point in self.waypoints:
                waypoint = rendering.make_circle(self.robot_radius/self.xyreso*scale_width)
                self.waypointtrans = rendering.Transform()
                waypoint.add_attr(self.waypointtrans)
                waypoint.set_color(0.8, 0.8, 0.8)
                self.waypointtrans.set_translation(point[0]/self.xyreso*scale_width, 
                        point[1]/self.xyreso*scale_height)
                self.viewer.add_geom(waypoint)

            # robot pose
            self.robottrans = rendering.Transform()
            orientation = rendering.make_capsule(self.robot_radius/self.xyreso*scale_width, 2.0)
            self.orientationtrans = rendering.Transform()

            # start
            start = rendering.make_circle(self.robot_radius*2/self.xyreso*scale_width)
            self.starttrans = rendering.Transform()
            start.add_attr(self.starttrans)
            start.set_color(0.7, 0.7, 1.0)
            #self.viewer.add_geom(start)
            # goal
            goal = rendering.make_circle(self.robot_radius*2/self.xyreso*scale_width)
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            goal.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(goal)

        self.starttrans.set_translation(self.start_p[0]/self.xyreso*scale_width, self.start_p[1]/self.xyreso*scale_height)
        self.goaltrans.set_translation(self.goal[0]/self.xyreso*scale_width, self.goal[1]/self.xyreso*scale_height)

        #robot
        robot_x = self.state[0]/self.xyreso * scale_width
        robot_y = self.state[1]/self.xyreso * scale_height
        self.robottrans.set_translation(robot_x, robot_y)
        self.orientationtrans.set_translation(robot_x, robot_y)
        self.orientationtrans.set_rotation(self.state[2])
        # robot
        robot = rendering.make_circle(self.human_radius/self.xyreso*scale_width)
        self.robottrans = rendering.Transform()
        robot.add_attr(self.robottrans)
        robot.set_color(0.0, 0.0, 1.0)
        self.robottrans.set_translation(robot_x, robot_y)
        self.viewer.add_onetime(robot)

        # human
        for i in range(len(self.human_state)):
            human = rendering.make_circle(self.human_radius/self.xyreso*scale_width)
            self.humantrans = rendering.Transform()
            human.add_attr(self.humantrans)
            human.set_color(0.2, 0.8, 0.2)
            
            target = rendering.make_circle(self.human_radius/self.xyreso*scale_width)
            self.targettrans = rendering.Transform()
            target.add_attr(self.targettrans)
            target.set_color(0.1,1.0,0.1)
            
            if human_mode < 5:
                self.humantrans.set_translation(self.human_state[i][0]/self.xyreso*scale_width, self.human_state[i][1]/self.xyreso*scale_height)
            else:
                self.humantrans.set_translation(self.sim.getAgentPosition(self.human_state[i])[0]/self.xyreso*scale_width, self.sim.getAgentPosition(self.human_state[i])[1]/self.xyreso*scale_height)
                self.targettrans.set_translation(self.target_position[i][0]/self.xyreso*scale_width, self.target_position[i][1]/self.xyreso*scale_height)
            self.viewer.add_onetime(human)
            if target_color:
                pass#self.viewer.add_onetime(target)
        # lidar
        if self.vis_lidar:
            for lidar in self.lidar:
               scan = rendering.make_capsule(lidar[1]/self.xyreso*scale_width, 2.0)
               self.scantrans= rendering.Transform()
               scan.add_attr(self.scantrans)
               if lidar[3]==1:
                   scan.set_color(1.0, 0.5, 0.5)#赤
               elif lidar[0]==0:#正面
                   scan.set_color(0.1, 1.0, 0.1)#緑
               else:
                   scan.set_color(0.0, 1.0, 1.0)
               self.scantrans.set_translation(robot_x, robot_y)
               self.scantrans.set_rotation(self.state[2]+lidar[0])
               self.viewer.add_onetime(scan)
            

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

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
