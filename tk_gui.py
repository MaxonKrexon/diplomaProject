from tkinter import *

import numpy as np
from PIL import Image as pilImage
from PIL import ImageTk, ImageOps
from PIL.Image import Resampling
import tkinter.simpledialog as sd
from startMission import *
from getData import *

from addTaskWindow import *

class Window:
    def __init__(self, width=100, height=100, title="New window", resizable=(False, False), iconPath=None):
        self.root = Tk()
        self.root.title(title)
        self.root.state('zoomed')
        self.root.resizable(resizable[0], resizable[1])
        if iconPath is not None:
            self.root.iconbitmap(iconPath)

        self.task = ''

    def run(self):
        self.drawWidgets()
        self.root.mainloop()

    def drawWidgets(self):

        self.mapFrame = LabelFrame(self.root, text="Карта местности", font="Arial 12")
        self.mapFrame.grid(row=0, rowspan=3, column=0, columnspan=4, sticky=N + S + W + E, padx=10)

        self.mapFile = pilImage.open(r"FIRMS_24hrs[@103.3,54.7,12z].jpg")
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.img = ImageTk.PhotoImage(self.mapFile.resize((round(0.75*w),round(0.75*h)), Resampling.LANCZOS))


        self.map = Canvas(self.mapFrame, width=round(0.75 * w), height=round(0.75 * h))
        self.map.create_image(0, 0, image = self.img, anchor=NW)
        self.map.pack()

        keyFrame = LabelFrame(self.root, text="Инструменты", font="Arial 12")
        keyFrame.grid(row=3, rowspan=3, column=0, columnspan=4, sticky=N + S + W + E, padx=10, pady=10)
        Button(keyFrame, text="Создать задание", command = self.addTask).pack(side=LEFT, padx=5, pady=5)
        Button(keyFrame, text="Задать точку", command= self.point).pack(side=LEFT, padx=5, pady=5)
        Button(keyFrame, text="Обновить", command=self.update).pack(side=LEFT, padx=5, pady=5)
        Button(keyFrame, text="Старт", command=self.start).pack(side=LEFT, padx=5, pady=5)
        Button(keyFrame, text="Просмотр данных", command=self.viewData).pack(side=LEFT, padx=5, pady=5)
        Button(keyFrame, text="Выход", command=self.root.destroy).pack(side=RIGHT, padx=5, pady=5)

        dronesFrame = LabelFrame(self.root, text="Список БПЛА", font="Arial 12")
        dronesFrame.grid(row=0, rowspan=2, column=4, columnspan=3, sticky=N + S + W + E, padx=10)
        anount = np.random.randint(2,3)
        self.drones = []
        for j in range(anount):
            positionX = np.random.randint(100, round(0.75*w) - 100)
            positionY = np.random.randint(100, round(0.75*h) - 100)
            self.drones.append([f"БПЛА {j+1}", IntVar(),positionX, positionY])

        self.drawDrones()


        for name, parameter,_,_ in self.drones:
            coin = np.random.randint(0,2)
            state = ["Готов", "Заряжается", "Неисправен"]
            reserve = np.random.randint(0,23)
            Checkbutton(dronesFrame, text = f"{name}\nСостояние: {state[coin]}\nЗапас хода: {reserve} мин",variable=parameter, justify=LEFT,relief=RAISED).pack(fill = X, padx = 5, pady = 5)

        feedbackFrame = LabelFrame(self.root, text="Обратная связь", font="Arial 12")
        feedbackFrame.grid(row=2, rowspan=4, column=4, columnspan=3, sticky=N + S + W + E, padx=10, pady=10)

        self.feedbackText = "Ожидание заданий..."
        self.text = Label(feedbackFrame, text=self.feedbackText)
        self.text.pack(pady= 5, padx=5)

    def addTask(self):
        filename = 'taskInfo.txt'
        taskInfo = open(filename,'w',encoding="UTF-8")
        taskInfo.close()
        addTaskWindow(self.root,filename, self.mapFile)


    def point(self):
        pass

    def viewData(self):
        getData(self.root,self.mapFile)


    def start(self):
        base = []
        name = 0
        state = 1
        posX = 2
        posY = 3
        
        for j in range(len(self.drones)):
            if self.drones[j][state].get() == 1:
                base.append([self.drones[j][posX],self.drones[j][posY]])


        startMission(base)


    def drawDrones(self):
        radius = 7
        for j in range(len(self.drones)):
            self.map.create_oval(self.drones[j][2] - radius, self.drones[j][3] + radius, self.drones[j][2] + radius, self.drones[j][3] - radius, fill='cyan')

    def update(self):
        x = 0
        y = 1
        file = open('coordinates.txt','r',encoding="UTF-8")
        coords = file.readlines()
        file.close()
        file = open('taskInfo.txt','r',encoding="UTF-8")
        setup = file.readlines()
        file.close()

        # Получение из файла задачи следующих данных
        # Реальные координаты левого верхнего угла карты
        topLeft = np.array([np.float32(setup[0].split(" ")[1]), np.float32(setup[0].split(" ")[2][:-1])])
        # Реальные координаты правого нижнего угла карты
        bottomRight = np.array([np.float32(setup[1].split(" ")[1]), np.float32(setup[1].split(" ")[2][:-1])])
        # Локальные координаты левого верхнего угла (значение пикселей на изображении)
        topLeftLocal = np.array([int(setup[3].split(" ")[1]), int(setup[3].split(" ")[2][:-1])])
        # Локальные координаты правого нижнего угла (значение пикселей на изображении)
        bottomRightLocal = np.array([int(setup[4].split(" ")[1]), int(setup[4].split(" ")[2][:-1])])

        north = topLeft[0]
        south = bottomRight[0]
        west = topLeft[1]
        east = bottomRight[1]

        x_width = abs((111.3*np.cos(np.deg2rad((north+south)/2))) * (west-east)) # km
        y_width = abs(111.3 * (north-south)) # km
        x_pixel_value = x_width/abs(topLeftLocal[x] - bottomRightLocal[x]) # km - значение 1 px в км
        y_pixel_value = y_width/abs(topLeftLocal[y] - bottomRightLocal[y]) # km

        # bottomRightLocal[1] - (int(lines[7+j].split(",")[y]) - topLeftLocal[1])

        path = np.zeros((2, np.size(coords)), int)
        for j in range(np.size(coords)):
            path[x][j] = topLeftLocal[0] + (float(coords[j].split(" ")[x]) / x_pixel_value)
            path[y][j] = bottomRightLocal[1] - ((float(coords[j].split(" ")[y][:-1]) / y_pixel_value) - topLeftLocal[1])
            # path[y][j] = bottomRightLocal[1] - ((float(coords[j].split(" ")[y][:-1]) / y_pixel_value) + topLeftLocal[1])

        for j in range(np.size(coords)-1):
            radius = 2
            self.map.create_oval(path[x][j] - radius,path[y][j] + radius, path[x][j]+radius, path[y][j] - radius, fill="yellow", outline="yellow")


if __name__ == "__main__":
    main = Window(title="Оператор", resizable=(True, True))
    main.run()
