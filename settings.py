import pickle
import numpy as np
import yaml

import os,sys
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

from PIL import Image, ImageTk
import cv2

class config_data():
    def __init__(self):
        self.map_data = None
        self.map_height = 0
        self.map_width = 0
        self.waypoints = []
        self.human_waypoints = []
        self.human_spawn_points = []
        self.robot_start_points = []
        self.goal_points = []

def image_set(file_name):
    img = cv2.imread(file_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = 240
    ret, img_thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

    #cv2.imshow('image',img_thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    map_data = np.where(img_thresh>100, 0, 1)

    return map_data

class settings_gui(Tk):
    def __init__(self):
        super().__init__()
        adjust_row = 10
        
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
    
        # Frame input map reso
        self.frame_reso = ttk.Frame(padding=5)
        self.frame_reso.grid(row=0, column=0, sticky=W+E)
        self.label_reso_str = ttk.Label(self.frame_reso, text="map resolution:")
        self.label_reso_str.pack(side=LEFT)
        self.entry_reso = StringVar()
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
    
        # Frame open waypoint
        adjust_row += 1
        self.frame_wayp = ttk.Frame(padding=5)
        self.frame_wayp.grid(row=adjust_row, column=0, sticky=W+E)
        self.label_wayp = ttk.Label(self.frame_wayp, text="waypoints file:")
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

        # Frame new waypoint create
        adjust_row += 1
        self.frame_nwyp = ttk.Frame(padding=5)
        self.frame_nwyp.grid(row=adjust_row, column=0, sticky=W+E)
        self.label_nwyp = ttk.Label(self.frame_nwyp, text="new waypoints name:")
        self.label_nwyp.pack(side=LEFT)
        self.e_nwyp = StringVar()
        self.entry_nwyp = ttk.Entry(self.frame_nwyp,
                textvariable=self.e_nwyp,
                width=50-len(self.label_nwyp.cget("text")))
        self.entry_nwyp.pack(side=LEFT)
        self.button_save = ttk.Button(self.frame_nwyp,
                text="save",
                command=self.save)
        self.button_save['state'] = DISABLED
        self.button_save.pack(padx = 3,side=RIGHT)
        self.button_create = ttk.Button(self.frame_nwyp,
                text="create",
                command=self.create)
        self.button_create.pack(padx = 3,side=RIGHT)

        # Frame display text
        adjust_row += 1
        separator = ttk.Separator(orient="horizontal")
        separator.grid(row=adjust_row, column=0, sticky="ew")
        adjust_row += 1
        self.frame_disp = ttk.Frame(padding=5)
        self.frame_disp.grid(row=adjust_row, column=0, sticky=W+E)
        self.label_disp = ttk.Label(self.frame_disp, text="<output>")
        self.label_disp.pack(side=LEFT)
        adjust_row += 1
        separator = ttk.Separator(orient="horizontal")
        separator.grid(row=adjust_row, column=0, sticky="ew")
        adjust_row += 1
        style = ttk.Style()
        style.configure("console.TFrame", background="white")#, font=20, anchor="w")
        self.frame_disp = ttk.Frame(padding=5,style="console.TFrame")
        self.frame_disp.grid(row=adjust_row, column=0, sticky=W+E)
        self.label_disp = ttk.Label(self.frame_disp, text="_", background="white")
        self.label_disp.pack(side=LEFT)

        # キャンセルボタンの設置
        #button_wayp = ttk.Button(frame_nwyp, text=("閉じる"), command=quit)
        #button_wayp.pack(fill = "x", padx=30, side = "left")

        self.map_canvas_obj= None
        self.map_canvas.create_text(self.canvas_width//2, self.canvas_height//2,
                text="--no map--", font=("", "30", "bold"))

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
            self.img = Image.open(open(file_path, 'rb'))
            self.width, self.height = self.img.size
            self.img = self.img.resize((self.canvas_width, self.canvas_height))
            self.map_image = ImageTk.PhotoImage(self.img)
            if self.map_canvas_obj is not None:
                    self.map_canvas.delete(self.map_canvas_obj)
            self.map_canvas_obj = self.map_canvas.create_image(
                    self.canvas_width/2,
                    self.canvas_height/2,
                    image=self.map_image)

    def waypointsfile_dialog(self):
        format_name = ".bin"
        dir_name = "maps/waypoints/"
        fTyp = [("waypoints binary file",format_name)]
        file_path = filedialog.askopenfilename(
                title = "open waypoints binary file",
                filetypes = fTyp,
                initialdir = "./"+dir_name)
        if len(file_path):
            self.entry_wayp.set(file_path)

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
            self.entry_reso_obj.insert(END,map_yaml['resolution'])
        else:
            self.label_disp["text"] = "yaml file has no resolution"
            self.label_disp["foreground"] = "yellow"


    def click_canvas(self, event):
        reso = float(self.entry_reso_obj.get())
        click_x = event.x*self.width/self.canvas_width*reso
        click_y = (self.canvas_height-event.y)*self.height/self.canvas_height*reso
        r = 2
        self.map_canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r)
        self.waypoints.append([click_x,click_y])

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
        self.entry_reso_obj.configure(state=NORMAL)
        self.label_disp['text'] = ""
        self.map_canvas.unbind('<Button-1>')
        self.waypoints = np.array(self.waypoints)
        with open("./maps/waypoints/"+file_name+'.bin', mode='wb') as f:
            pickle.dump(self.waypoints, f)

    def create(self):
        if self.width == 0:
            self.label_disp['text'] = "load map image"
            self.label_disp['foreground'] = "red"
            return False
        try:
            reso = float(self.entry_reso_obj.get())
            if reso <= 0:
               raise Exception
        except:
            self.label_disp['text'] = "correct resolution value"
            self.label_disp['foreground'] = "red"
            return False
        self.label_disp['text'] = "click waypoints"
        self.label_disp['foreground'] = "blue"
        self.button_save["state"] = NORMAL
        self.entry_reso_obj.configure(state=DISABLED)
        self.map_canvas.bind('<Button-1>', self.click_canvas)
        self.waypoints = []


if __name__ == "__main__":
    gui = settings_gui()
    # rootの作成
    gui.title("SS2D SETTING GUI")
    gui.mainloop()
