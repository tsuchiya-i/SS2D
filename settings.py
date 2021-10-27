import pickle
import numpy as np
import cv2

import os,sys
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

from PIL import Image, ImageTk

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
    
        #画像設置
        self.canvas_width = 300
        self.canvas_height = 200
        self.geometry("800x400")

        self.map_canvas = Canvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="gray"
        )
        self.map_canvas.grid(rowspan=10, row=0, column=0, sticky=W+N)
    
        # Frame1の作成
        self.frame1 = ttk.Frame(padding=10)
        self.frame1.grid(row=10, column=0, sticky=E)
    
        # 「ファイル参照」ラベルの作成
        self.IFileLabel = ttk.Label(self.frame1, text="file:", padding=(5, 2))
        self.IFileLabel.pack(side=LEFT)
    
        # 「ファイル参照」エントリーの作成
        self.entry2 = StringVar()
        self.IFileEntry = ttk.Entry(self.frame1, textvariable=self.entry2, width=30)
        self.IFileEntry.pack(side=LEFT)
    
        # 「ファイル参照」ボタンの作成
        self.IFileButton = ttk.Button(self.frame1, text="参照", command=self.filedialog_clicked)
        self.IFileButton.pack(side=LEFT)

        self.map_canvas_obj= None
    
        """
        # Frame3の作成
        frame3 = ttk.Frame(root, padding=10)
        frame3.grid(row=5,column=0,sticky=W)
    
        # 実行ボタンの設置
        button1 = ttk.Button(frame3, text="実行", command=conductMain)
        button1.pack(fill = "x", padx=30, side = "left")
    
        # キャンセルボタンの設置
        button2 = ttk.Button(frame3, text=("閉じる"), command=quit)
        button2.pack(fill = "x", padx=30, side = "left")
        """

        # フォルダ指定の関数
    def dirdialog_clicked(self):
        iDir = os.path.abspath(os.path.dirname(__file__))
        iDirPath = filedialog.askdirectory(initialdir = iDir)
        self.entry1.set(iDirPath)
    
    # ファイル指定の関数
    def filedialog_clicked(self):
        fTyp = [("Image file",".bmp .png .jpg .tif"), ("Bitmap",".bmp"), ("PNG",".png"), ("JPEG",".jpg"), ("Tiff",".tif")]
        file_path = filedialog.askopenfilename(
                title = "open map image",
                filetypes = fTyp,
                initialdir = "./maps/")
        self.entry2.set(file_path)
    
        #print(filename)
        #print(type(filename))
        #IFileButton["text"] = "aaa"
    
        self.img = Image.open(open(file_path, 'rb'))
        self.img = self.img.resize((self.canvas_width // 2, self.canvas_height // 2))
        #self.img.show()
        self.map_image = ImageTk.PhotoImage(self.img)
        
        if self.map_canvas_obj is not None:
                self.map_canvas.delete(self.map_canvas_obj)
        self.map_canvas_obj = self.map_canvas.create_image(self.canvas_width/2, self.canvas_height/2, image=self.map_image)
    
    # 実行ボタン押下時の実行関数
    def conductMain(self):
        text = ""
    
        dirPath = self.entry1.get()
        filePath = self.entry2.get()
        if dirPath:
            text += "フォルダパス：" + dirPath + "\n"
        if filePath:
            text += "ファイルパス：" + filePath
    
        if text:
            messagebox.showinfo("info", text)
        else:
            messagebox.showerror("error", "パスの指定がありません。")


if __name__ == "__main__":
    gui = settings_gui()
    # rootの作成
    #gui.title("window name")
    gui.mainloop()
