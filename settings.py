import pickle
import numpy as np
import yaml

import os,sys
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk
import cv2

class config_data():
    def __init__(self):
        self.map_data = None
        self.map_height = 0
        self.map_width = 0
        self.nwaypoints = []
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
        self.nwaypoints = []
        self.waypoints = []
        
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
        self.e_nwyp = StringVar()
        self.entry_nwyp = ttk.Entry(self.frame_nwyp,
                textvariable=self.e_nwyp,
                width=41-len(self.label_nwyp.cget("text")))
        self.entry_nwyp.pack(side=RIGHT)

        # Frame display text
        adjust_row += 1
        separator = ttk.Separator(orient="horizontal")
        separator.grid(row=adjust_row, column=0, sticky="ew")
        adjust_row += 1

        self.frame_disp = ttk.Frame(padding=5)
        self.frame_disp.grid(row=adjust_row, column=0, sticky=W+E)
        self.scr_disp = ScrolledText(self.frame_disp, font=("",15), height=4,width=45)
        self.scr_disp.pack()
        self.insert_disp_text("> ")

        #self.scr_disp = ttk.Label(self.frame_disp, text="_", background="white")
        #self.scr_disp.pack(side=LEFT)

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
        format_name = ".bin"
        dir_name = "maps/waypoints/"
        fTyp = [("waypoints binary file",format_name)]
        file_path = filedialog.askopenfilename(
                title = "open waypoints binary file",
                filetypes = fTyp,
                initialdir = "./"+dir_name)
        if len(file_path):
            self.entry_wayp.set(file_path)
            with open(file_path, mode='rb') as f:
                self.waypoints = pickle.load(f)
            self.oval_remove(self.waypoints)
            self.oval_draw(reso,self.waypoints)

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
        click_y = (self.canvas_height-event.y)*self.height/self.canvas_height*reso
        r = 2
        self.map_canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r,
                fill="cyan", tag="new_wyp"+str(len(self.nwaypoints)))
        self.nwaypoints.append([click_x,click_y])
        self.insert_disp_text(str([round(click_x,3),round(click_y,3)])+"\n")

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
        self.insert_disp_text("   Saved.\n> ")
        self.map_canvas.unbind('<Button-1>')
        with open("./maps/waypoints/"+file_name+'.bin', mode='wb') as f:
            pickle.dump(self.nwaypoints, f)
        self.oval_remove(self.nwaypoints)
        self.nwaypoints = []

    def create(self):
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
        self.insert_disp_text("click points\n")
        self.button_save["state"] = NORMAL
        self.button_create["state"] = DISABLED
        self.button_back["state"] = NORMAL
        self.entry_reso_obj.configure(state=DISABLED)
        self.map_canvas.bind('<Button-1>', self.click_canvas)
        self.oval_remove(self.waypoints)

    def back(self):
        if len(self.nwaypoints) == 0:
            self.button_save["state"] = DISABLED
            self.button_create["state"] = NORMAL
            self.button_back["state"] = DISABLED
            self.map_canvas.unbind('<Button-1>')
            self.insert_disp_text("  Canceled.\n> ")
            return True
        self.nwaypoints.pop()
        self.delete_disp_text('end -2lines linestart', END)
        self.insert_disp_text("\n")
        self.map_canvas.delete("new_wyp"+str(len(self.nwaypoints)))

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
    def oval_draw(self,reso,points):
        tag_name = self.tag_name_select(points)
        r = 2
        for i, xy in enumerate(self.waypoints):
            x = xy[0]/(self.width/self.canvas_width*reso)
            y = self.canvas_height - xy[1]/(self.height/self.canvas_height*reso)
            self.map_canvas.create_oval(x-r, y-r, x+r, y+r,
                    fill="cyan", tag=tag_name+str(i))
    def tag_name_select(self,points):
        if self.nwaypoints == points:
            tag_name = "new_wyp"
        elif self.waypoints == points:
            tag_name = "wyp"
        return tag_name

if __name__ == "__main__":
    gui = settings_gui()
    # rootの作成
    gui.title("SS2D SETTING GUI")
    gui.mainloop()
