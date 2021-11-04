import pickle
import numpy as np
import yaml
import os,sys,io
import threading
import json

from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk
import cv2

import gym
import ss2d

class config_data():
    def __init__(self):
        self.thresh_map = None #image array(2D)
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
        self.wall_render = False #bool
        self.console_output = True #bool


def image_thresh(file_name):
    img = cv2.imread(file_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = 240
    ret, img_thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('image',img_thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img_thresh

class settings_gui(Tk):
    def __init__(self):
        super().__init__()
        adjust_row = 10
        self.nwaypoints = []
        #保存パラメータ
        self.waypoints = []
        self.goal_points = []
        self.human_points = []
        self.reso = 0
        
        #画像設置
        self.canvas_width = 600
        self.canvas_height = 400
        self.width, self.height = (0,0)

        self.map_canvas = Canvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="gray"
        )
        self.map_canvas.grid(rowspan=adjust_row, row=1, column=0, sticky=W)
        self.map_canvas.unbind('<Button-1>')
        self.map_canvas_obj= None
        self.map_canvas.create_text(self.canvas_width//2, self.canvas_height//2,
                text="--no map--", font=("", "30", "bold"))
    
        # Frame input map reso
        self.frame_reso = ttk.Frame(padding=5)
        self.frame_reso.grid(row=0, column=0, sticky=W+E)
        self.label_reso_str = ttk.Label(self.frame_reso, text="map resolution:")
        self.label_reso_str.pack(side=LEFT)
        self.entry_reso = DoubleVar(value=0.05)
        self.entry_reso_obj = ttk.Entry(self.frame_reso,
                textvariable=self.entry_reso,
                width=8)
        self.entry_reso_obj.pack(side=LEFT)
        self.label_str = ttk.Label(self.frame_reso,
                text="(m/pixel)        Load yamlfile:",
                padding=(3, 2))
        self.label_str.pack(side=LEFT)
        self.entry_yaml = StringVar()
        self.entry_yaml_obj = ttk.Entry(self.frame_reso,
                textvariable=self.entry_yaml,
                width=18)
        self.entry_yaml_obj.pack(side=LEFT)
        self.botton_yaml = ttk.Button(self.frame_reso,
                text="Browse...",
                command=self.yamlfile_dialog)
        self.botton_yaml.pack(side=RIGHT)

        # Frame open map
        adjust_row += 1
        self.frame_map = ttk.Frame(padding=5)
        self.frame_map.grid(row=adjust_row, column=0, sticky=W+E)
        self.label_map = ttk.Label(self.frame_map, text="map image file:")
        self.label_map.pack(side=LEFT)
        self.button_map = ttk.Button(self.frame_map,
                text="Browse...",
                command=self.imagefile_dialog)
        self.button_map.pack(side=RIGHT)
        self.entry_map = StringVar()
        self.entry_map_obj = ttk.Entry(self.frame_map,
                textvariable=self.entry_map,
                width=45)
        self.entry_map_obj.pack(padx=5,side=RIGHT)
    
        # Frame open start points
        adjust_row += 1
        self.frame_wayp = ttk.Frame(padding=5)
        self.frame_wayp.grid(row=adjust_row, column=0, sticky=W+E)
        self.label_wayp = ttk.Label(self.frame_wayp, text="start points file:")
        self.label_wayp.pack(side=LEFT)
        self.button_wayp = ttk.Button(self.frame_wayp,
                text="Browse...",
                command=self.waypointsfile_dialog)
        self.button_wayp.pack(side=RIGHT)
        self.entry_wayp = StringVar()
        self.entry_wayp_obj = ttk.Entry(self.frame_wayp,
                textvariable=self.entry_wayp,
                width=45)
        self.entry_wayp_obj.pack(padx=5,side=RIGHT)

        # Frame open goal point
        adjust_row += 1
        self.frame_gp = ttk.Frame(padding=5)
        self.frame_gp.grid(row=adjust_row, column=0, sticky=W+E)
        self.label_gp = ttk.Label(self.frame_gp, text="goal points file:")
        self.label_gp.pack(side=LEFT)
        self.button_gp = ttk.Button(self.frame_gp,
                text="Browse...",
                command=self.goal_pointsfile_dialog)
        self.button_gp.pack(side=RIGHT)
        self.entry_gp = StringVar()
        self.entry_gp_obj = ttk.Entry(self.frame_gp,
                textvariable=self.entry_gp,
                width=45)
        self.entry_gp_obj.pack(padx=5,side=RIGHT)

        # Frame open human point
        adjust_row += 1
        self.frame_hp = ttk.Frame(padding=5)
        self.frame_hp.grid(row=adjust_row, column=0, sticky=W+E)
        self.label_hp = ttk.Label(self.frame_hp, text="human points file:")
        self.label_hp.pack(side=LEFT)
        self.button_hp = ttk.Button(self.frame_hp,
                text="Browse...",
                command=self.human_pointsfile_dialog)
        self.button_hp.pack(side=RIGHT)
        self.entry_hp = StringVar()
        self.entry_hp_obj = ttk.Entry(self.frame_hp,
                textvariable=self.entry_hp,
                width=45)
        self.entry_hp_obj.pack(padx=5,side=RIGHT)

        # Frame new waypoint create
        adjust_row += 1
        self.frame_nwyp = ttk.Frame(padding=5)
        self.frame_nwyp.grid(row=adjust_row, column=0, sticky=W+E)
        self.label_nwyp = ttk.Label(self.frame_nwyp, text="new waypoints name:")
        self.label_nwyp.pack(side=LEFT)
        self.button_save = ttk.Button(self.frame_nwyp,
                text="save",
                command=self.save)
        self.button_save['state'] = DISABLED
        self.button_save.pack(padx = 3,side=RIGHT)
        self.button_back = ttk.Button(self.frame_nwyp,
                text="back",
                command=self.back)
        self.button_back['state'] = DISABLED
        self.button_back.pack(padx = 3,side=RIGHT)
        self.button_create = ttk.Button(self.frame_nwyp,
                text="create",
                command=self.create)
        self.button_create.pack(padx = 5,side=RIGHT)
        self.entry_nwyp = StringVar()
        self.entry_nwyp_obj = ttk.Entry(self.frame_nwyp,
                textvariable=self.entry_nwyp,
                width=41-len(self.label_nwyp.cget("text")))
        self.entry_nwyp_obj.pack(side=RIGHT)



        #######################################
        ############ RIGHT ZONE ###############
        #######################################
        # world param
        R_adjust_row = 0
        width = 300
        self.frame_world = ttk.Labelframe(padding=5, width=width, height=50, text="World parameter")
        self.frame_world.propagate(False) 
        self.frame_world.grid(row=R_adjust_row, column=1, sticky=W)
        self.label_dt = ttk.Label(self.frame_world, text=" dt: ")
        self.label_dt.pack(side=LEFT)
        self.entry_dt = DoubleVar(value=0.1)
        self.entry_dt_obj = ttk.Entry(self.frame_world,
                textvariable=self.entry_dt,
                width=10)
        self.entry_dt_obj.pack(side=LEFT)
        self.label_dt = ttk.Label(self.frame_world, text=" (sec/step)  ")
        self.label_dt.pack(side=LEFT)
        # goal frame
        R_adjust_row += 1
        self.frame_goal = ttk.Labelframe(padding=5, width=width, height=75, text="Goal point option")
        self.frame_goal.propagate(False) 
        self.frame_goal.grid(row=R_adjust_row, column=1, sticky=W)
        self.goal_option = IntVar()
        rb1 = Radiobutton(self.frame_goal, text='random from goal points',
                value=0, variable=self.goal_option)
        rb1.pack(anchor=W)
        rb2 = Radiobutton(self.frame_goal, text='random from start points',
                value=1, variable=self.goal_option)
        rb2.pack(anchor=W)
        # robot param
        R_adjust_row += 1
        self.frame_robot = ttk.Labelframe(padding=5, width=width, height=100, text="Robot parameter")
        self.frame_robot.propagate(False) 
        self.frame_robot.grid(row=R_adjust_row, column=1, sticky=W+E)
        self.label_radius = ttk.Label(self.frame_robot, text=" Robot radius: ")
        self.label_radius.grid(row=0, column=0, sticky=W)
        self.entry_radius = DoubleVar(value=0.3)
        self.entry_radius_obj = ttk.Entry(self.frame_robot,
                textvariable=self.entry_radius,
                width=10)
        self.entry_radius_obj.grid(row=0, column=1, sticky=W)
        self.label_radius = ttk.Label(self.frame_robot, text=" (m)  ")
        self.label_radius.grid(row=0, column=2, sticky=W)
        self.label_ = ttk.Label(self.frame_robot, text=" Robot Lidar: ")
        self.label_.grid(row=1, column=0, sticky=W)
        self.label_ = ttk.Label(self.frame_robot, text="  max: ")
        self.label_.grid(row=2, column=0, sticky=E)
        self.entry_lmax = DoubleVar(value=10.0)
        self.entry_lmax_obj = ttk.Entry(self.frame_robot,
                textvariable=self.entry_lmax,
                width=10)
        self.entry_lmax_obj.grid(row=2, column=1, sticky=W)
        self.label_ = ttk.Label(self.frame_robot, text="(m)")
        self.label_.grid(row=2, column=2, sticky=W)
        self.label_ = ttk.Label(self.frame_robot, text="  min: ")
        self.label_.grid(row=3, column=0, sticky=E)
        self.entry_lmin = DoubleVar(value=0.2)
        self.entry_lmin_obj = ttk.Entry(self.frame_robot,
                textvariable=self.entry_lmin,
                width=10)
        self.entry_lmin_obj.grid(row=3, column=1, sticky=W)
        self.label_ = ttk.Label(self.frame_robot, text="(m)")
        self.label_.grid(row=3, column=2, sticky=W)
        self.label_ = ttk.Label(self.frame_robot, text="Viewing angle: ")
        self.label_.grid(row=4, column=0, sticky=E)
        self.entry_la = DoubleVar(value=90)
        self.entry_la_obj = ttk.Entry(self.frame_robot,
                textvariable=self.entry_la,
                width=10)
        self.entry_la_obj.grid(row=4, column=1, sticky=W)
        self.label_ = ttk.Label(self.frame_robot, text="(deg)")
        self.label_.grid(row=4, column=2, sticky=W)
        self.label_ = ttk.Label(self.frame_robot, text="Angle reso: ")
        self.label_.grid(row=5, column=0, sticky=E)
        self.entry_lar = DoubleVar(value=10)
        self.entry_lar_obj = ttk.Entry(self.frame_robot,
                textvariable=self.entry_lar,
                width=10)
        self.entry_lar_obj.grid(row=5, column=1, sticky=W)
        self.label_ = ttk.Label(self.frame_robot, text="(deg)")
        self.label_.grid(row=5, column=2, sticky=W)
        # human param
        R_adjust_row += 1
        self.frame_human = ttk.Labelframe(padding=5, width=width, height=100, text="Human parameter")
        self.frame_human.propagate(False)
        self.frame_human.grid(row=R_adjust_row, column=1, sticky=W+E)
        self.label_ = ttk.Label(self.frame_human, text="Number of people: ")
        self.label_.grid(row=1, column=0, sticky=W)
        self.entry_hn = IntVar(value=0)
        self.entry_hn_obj = ttk.Entry(self.frame_human,
                textvariable=self.entry_hn,
                width=10)
        self.entry_hn_obj.grid(row=1, column=1, sticky=W)
        self.label_ = ttk.Label(self.frame_human, text="(person)")
        self.label_.grid(row=1, column=2, sticky=E)
        self.label_ = ttk.Label(self.frame_human, text="Human spawn point: ")
        self.label_.grid(row=2, column=0, sticky=W)
        self.human_option = IntVar()
        rb1 = Radiobutton(self.frame_human, text='random from human spawn points',
                value='0', variable=self.human_option)
        rb1.grid(columnspan=3, row=3, column=0, sticky=W)
        rb2 = Radiobutton(self.frame_human, text='random from start points',
                value='1', variable=self.human_option)
        rb2.grid(columnspan=3, row=4, column=0, sticky=W)
        # observe option
        R_adjust_row += 1
        self.frame_observe = ttk.Labelframe(padding=5, width=width, height=100, text="Observe option")
        self.frame_observe.propagate(False)
        self.frame_observe.grid(row=R_adjust_row, column=1, sticky=W+E)
        self.detect_TF = BooleanVar()
        check = Checkbutton(self.frame_observe, text="Activate Human Detection",
                variable=self.detect_TF)
        check.grid(row=2, column=0, sticky=W)

        # Simulator param
        R_adjust_row += 1
        self.frame_sim = ttk.Labelframe(padding=5, width=width, height=200, text="Simulator option")
        self.frame_sim.propagate(False)
        self.frame_sim.grid(rowspan=8, row=R_adjust_row, column=1, sticky=W+E+N)
        self.wall_TF = BooleanVar()
        check = Checkbutton(self.frame_sim, text="Activate wall Rendering",
                variable=self.wall_TF)
        check.grid(row=1, column=0, sticky=W)
        self.output_TF = BooleanVar()
        check = Checkbutton(self.frame_sim, text="Console output(collision & goal)",
                variable=self.output_TF)
        check.grid(row=2, column=0, sticky=W)
        # test mode
        R_adjust_row += 8
        self.frame_test = ttk.Labelframe(padding=5, width=width, height=200, text="Test run mode")
        self.frame_test.propagate(False)
        self.frame_test.grid(rowspan=2, row=R_adjust_row, column=1, sticky=W+E+N)
        self.button_run = ttk.Button(self.frame_test,
                text="Run",
                command=self.run)
        self.button_run.grid(row=1, column=0, sticky=W)
        self.button_close = ttk.Button(self.frame_test,
                text="Close",
                command=self.close)
        self.button_close.grid(row=1, column=1, sticky=E)
        self.button_close["state"]=DISABLED

        # save config
        R_adjust_row += 2
        self.frame_save_config = ttk.Label(padding=5)
        self.frame_save_config.grid(row=R_adjust_row, column=1)
        self.button_save_config = ttk.Button(self.frame_save_config,
                text="Save",
                command=self.save_config,
                width=35)
        self.button_save_config.pack(side=TOP)

        ###################################
        ####### Frame display text#########
        ###################################
        #separator = ttk.Separator(orient="horizontal")
        #separator.grid(row=adjust_row, column=0, sticky="ew")
        adrow = max(R_adjust_row,adjust_row+1)
        self.frame_disp = ttk.Frame(padding=5)
        self.frame_disp.grid(rowspan=adrow, row=0, column=2)
        self.scr_disp = ScrolledText(self.frame_disp, font=("",10), width=30,height=35)
        self.scr_disp.pack()
        self.insert_disp_text("> ")

    # ファイル指定の関数
    def imagefile_dialog(self):
        format_name = ".bmp .BMP .png .PNG .jpg .JPG"
        dir_name = "maps/"
        fTyp = [("Image file",format_name)]
        file_path = filedialog.askopenfilename(
                title = "open map image",
                filetypes = fTyp,
                initialdir = "./"+dir_name)
        if len(file_path):
            self.entry_map.set(file_path)
            self.cv_thresh_image = image_thresh(file_path)
            self.img = Image.fromarray(self.cv_thresh_image)
            #self.img = Image.open(open(file_path, 'rb'))
            self.width, self.height = self.img.size
            self.img = self.img.resize((self.canvas_width, self.canvas_height))
            self.map_image = ImageTk.PhotoImage(self.img)
            if self.map_canvas_obj is not None:
                    self.map_canvas.delete(self.map_canvas_obj)
            self.map_canvas_obj = self.map_canvas.create_image(
                    self.canvas_width/2,
                    self.canvas_height/2,
                    image=self.map_image)
            self.oval_remove(self.waypoints)
            self.waypoints = []

    def waypointsfile_dialog(self):
        if self.map_reso_check():
            self.reso = float(self.entry_reso_obj.get())
            format_name = ".bin"
            dir_name = "maps/waypoints/"
            fTyp = [("waypoints binary file",format_name)]
            file_path = filedialog.askopenfilename(
                    title = "open waypoints binary file",
                    filetypes = fTyp,
                    initialdir = "./"+dir_name)
            if len(file_path):
                self.entry_wayp.set(file_path)
                self.oval_remove(self.waypoints)
                with open(file_path, mode='rb') as f:
                    self.waypoints = pickle.load(f)
                self.oval_draw(self.reso,self.waypoints)

    def goal_pointsfile_dialog(self):
        if self.map_reso_check():
            self.reso = float(self.entry_reso_obj.get())
            format_name = ".bin"
            dir_name = "maps/waypoints/"
            fTyp = [("waypoints binary file",format_name)]
            file_path = filedialog.askopenfilename(
                    title = "open waypoints binary file",
                    filetypes = fTyp,
                    initialdir = "./"+dir_name)
            if len(file_path):
                self.entry_gp.set(file_path)
                self.oval_remove(self.goal_points)
                with open(file_path, mode='rb') as f:
                    self.goal_points = pickle.load(f)
                self.oval_draw(self.reso,self.goal_points,"red")

    def human_pointsfile_dialog(self):
        if self.map_reso_check():
            self.reso = float(self.entry_reso_obj.get())
            format_name = ".bin"
            dir_name = "maps/waypoints/"
            fTyp = [("waypoints binary file",format_name)]
            file_path = filedialog.askopenfilename(
                    title = "open waypoints binary file",
                    filetypes = fTyp,
                    initialdir = "./"+dir_name)
            if len(file_path):
                self.entry_hp.set(file_path)
                self.oval_remove(self.human_points)
                with open(file_path, mode='rb') as f:
                    self.human_points = pickle.load(f)
                self.oval_draw(self.reso,self.human_points,"orange")

    def yamlfile_dialog(self):
        format_name = ".yaml"
        dir_name = "maps/"
        fTyp = [("map info yaml file",format_name)]
        file_path = filedialog.askopenfilename(
                title = "open map info yaml file",
                filetypes = fTyp,
                initialdir = "./"+dir_name)
        if len(file_path):
            self.entry_yaml.set(file_path)
            with open(file_path, 'r') as yml:
                map_yaml = yaml.load(yml,Loader=yaml.SafeLoader)
            if 'resolution' in map_yaml:
                self.entry_reso_obj.delete(0, END)
                self.entry_reso_obj.insert(END,map_yaml['resolution'])
            else:
                messagebox.showwarning('Error', "This yaml has no resolution value.")

    def click_canvas(self, event):
        reso = float(self.entry_reso_obj.get())
        click_x = event.x*self.width/self.canvas_width*reso
        click_x = round(click_x,3)
        click_y = (self.canvas_height-event.y)*self.height/self.canvas_height*reso
        click_y = round(click_y,3)
        r = 2
        self.map_canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r,
                fill="cyan", tag="new_wyp"+str(len(self.nwaypoints)))
        self.nwaypoints.append([click_x,click_y])
        self.insert_disp_text(str([click_x,click_y])+"\n")

    def save(self):
        if self.entry_nwyp.get() == "":
            messagebox.showwarning('Error', "No file name entered.")
            return False
        else:
            file_name = self.entry_nwyp.get()
            if file_name[-4:]==".bin" or file_name[-4:]==".BIN":
                file_name = file_name[:-4]
            if os.path.exists('./maps/waypoints/'+file_name+'.bin'):
                yn = messagebox.askquestion("Confirm","Overwrite file "+file_name+'.bin ?')
                if yn == "no":
                    return False
        self.button_save["state"] = DISABLED
        self.button_create["state"] = NORMAL
        self.button_back["state"] = DISABLED
        self.entry_reso_obj.configure(state=NORMAL)
        self.map_canvas.unbind('<Button-1>')
        with open("./maps/waypoints/"+file_name+'.bin', mode='wb') as f:
            pickle.dump(self.nwaypoints, f)
        self.insert_disp_text("   "+file_name+".bin Saved.\n("+str(len(self.nwaypoints))+" points)\n> ")
        self.entry_nwyp_obj.delete(0,END)
        self.oval_remove(self.nwaypoints)
        self.nwaypoints = []
        self.oval_draw(self.reso,self.waypoints)
        self.oval_draw(self.reso,self.goal_points)
        self.oval_draw(self.reso,self.human_points)

    def create(self):
        if self.map_reso_check():
            self.insert_disp_text("click points\n")
            self.button_save["state"] = NORMAL
            self.button_create["state"] = DISABLED
            self.button_back["state"] = NORMAL
            self.entry_reso_obj.configure(state=DISABLED)
            self.map_canvas.bind('<Button-1>', self.click_canvas)
            self.oval_remove(self.waypoints)
            self.oval_remove(self.goal_points)
            self.oval_remove(self.human_points)

    def back(self):
        if len(self.nwaypoints) == 0:
            self.button_save["state"] = DISABLED
            self.button_create["state"] = NORMAL
            self.button_back["state"] = DISABLED
            self.map_canvas.unbind('<Button-1>')
            self.insert_disp_text("  Canceled.\n> ")
            self.entry_reso_obj.configure(state=NORMAL)
            self.oval_draw(self.reso,self.waypoints)
            self.oval_draw(self.reso,self.goal_points,"red")
            self.oval_draw(self.reso,self.human_points,"orange")
            return True
        self.nwaypoints.pop()
        self.delete_disp_text('end -2lines linestart', END)
        self.insert_disp_text("\n")
        self.map_canvas.delete("new_wyp"+str(len(self.nwaypoints)))

    def run(self):
        thread = threading.Thread(target=self.run_simulator)
        thread.start()
        self.button_run["state"] = DISABLED
        self.button_close["state"] = NORMAL

    def close(self):
        self.run_flag = False
        self.button_run["state"] = NORMAL
        self.button_close["state"] = DISABLED

    def run_simulator(self):
        self.run_flag = True
        self.env = gym.make('ss2d-v0')
        observation = self.env.reset()
        while self.run_flag:
            self.env.render()
            action = self.env.action_space.sample()
            observation, reward, done,  _ = self.env.step(action)
            if done:
                self.env.reset() #衝突ORゴールで環境リセット
        self.env.close()
        
    def save_config(self):
        save_data = config_data()
        save_data.thresh_map = self.cv_thresh_image #[[],[],,,]
        save_data.start_points = self.waypoints #(m,m)
        if self.goal_option.get()==0:
            save_data.goal_points = self.goal_points #####
        elif self.goal_option.get()==1:
            save_data.goal_points = self.waypoints #####
        if self.human_option.get()==0:
            save_data.human_points = self.human_points
        elif self.human_option.get()==1:
            save_data.human_points = self.waypoints
        save_data.reso = self.entry_reso.get() #m/pix
        save_data.world_dt = self.entry_dt.get() #sec
        save_data.robot_r = self.entry_radius.get() #m
        save_data.lidar_max = self.entry_lmax.get() #m
        save_data.lidar_min = self.entry_lmin.get() #m
        save_data.lidar_angle = self.entry_la.get() #deg
        save_data.lidar_reso = self.entry_lar.get() #deg
        save_data.human_n = self.entry_hn.get() #person
        save_data.human_detect = self.detect_TF.get() #bool
        save_data.wall_render = self.wall_TF.get() #bool
        save_data.console_output = self.output_TF.get() #bool
        
        with io.StringIO() as f:
            sys.stdout = f
            print(vars(save_data))
            text = f.getvalue().replace(", '",",\n'")
            text = text.replace(", [",",\n[")

        with open("./ss2d/envs/config.bin", mode='wb') as f:
            pickle.dump(save_data, f)
        self.insert_disp_text(text+"Setting Saved.\n\n>")

        
    def delete_disp_text(self,start,end):
        self.scr_disp["state"] = NORMAL
        self.scr_disp.delete(start,end)
        self.scr_disp.see("end")
        self.scr_disp["state"] = DISABLED
    def insert_disp_text(self,text):
        self.scr_disp["state"] = NORMAL
        self.scr_disp.insert("end",text)
        self.scr_disp.see("end")
        self.scr_disp["state"] = DISABLED
    def oval_remove(self,points):
        tag_name = self.tag_name_select(points)
        for i in range(len(points)):
            self.map_canvas.delete(tag_name+str(i))
    def oval_draw(self,reso,points,color="cyan"):
        tag_name = self.tag_name_select(points)
        r = 2
        for i, xy in enumerate(points):
            x = xy[0]/(self.width/self.canvas_width*reso)
            y = self.canvas_height - xy[1]/(self.height/self.canvas_height*reso)
            self.map_canvas.create_oval(x-r, y-r, x+r, y+r,
                    fill=color, tag=tag_name+str(i))
    def tag_name_select(self,points):
        if self.nwaypoints == points:
            tag_name = "new_wyp"
        elif self.waypoints == points:
            tag_name = "wyp"
        elif self.goal_points == points:
            tag_name = "goal_wyp"
        elif self.human_points == points:
            tag_name = "human_wyp"

        return tag_name

    def map_reso_check(self):
        if self.width == 0:
            messagebox.showwarning('Error', "Load map image.")
            return False
        try:
            reso = float(self.entry_reso_obj.get())
            if reso <= 0:
               raise Exception
        except:
            messagebox.showwarning('Error', "correct resolution value.")
            return False
        return True

if __name__ == "__main__":
    gui = settings_gui()
    # rootの作成
    gui.title("SS2D SETTING GUI")
    gui.mainloop()
