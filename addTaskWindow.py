import math
import time
from tkinter import *
from tkinter.ttk import Combobox
import tkinter.messagebox as mb
from PIL import Image as pilImage
from PIL import ImageTk, ImageOps
from PIL.Image import Resampling
from askForCoords import *

class addTaskWindow:
    def __init__(self, parent, filename, map, title="Создать задачу", resizable=(False, False), iconPath=None):
        self.child = Toplevel(parent)
        self.child.title(title)
        self.filename = filename
        self.child.state('normal')
        self.file = open(self.filename,'a',encoding="UTF-8")
        self.child.grab_set()
        self.child.focus_set()
        self.child.resizable(resizable[0], resizable[1])
        self.choice = IntVar()
        self.shapeType = "Single"
        self.mapFile = map
        self.nextStep = ''
        self.objects = []
        self.logfile = open('log.txt','w',encoding="UTF-8")

        if iconPath is not None:
            self.child.iconbitmap(iconPath)

        self.drawWidgets()

    def drawWidgets(self):
        formationFrame = LabelFrame(self.child, text="Групповое выполнение")
        formationFrame.grid(column=0,columnspan=1,row=0,rowspan=1)

        Checkbutton(formationFrame, text="Ипользовать построение", variable=self.choice).pack(padx= 5, pady=5, anchor=W)
        Label(formationFrame, text="Тип построения", justify=LEFT).pack(padx= 5, pady=5, anchor=W)

        self.shape = Combobox(formationFrame, values=("Колонна", "Шеренга", "Квадрат"), justify=LEFT)
        self.shape.pack(padx= 5, pady=5, fill = X)

        taskFrame = LabelFrame(self.child, text="Задание")
        taskFrame.grid(column=0,columnspan=1,row=1,rowspan=2)
        Button(taskFrame,text="Выделить область работы", command=self.setRegion).pack(pady=5,padx=5, fill=X)
        Button(taskFrame, text="Задать объект исследования",command=self.setObject).pack(pady=5,padx=5,fill=X)
        Button(taskFrame, text="Задать точку сбора",command=self.setGatheringPoint).pack(pady=5,padx=5,fill=X)
        Button(taskFrame, text="Сохранить", command= self.saveData).pack(pady=5,padx=5,fill=X)


        # mapFrame = LabelFrame(self.child, text="Карта местности")

        w, h = self.child.winfo_screenwidth(), self.child.winfo_screenheight()
        self.img = ImageTk.PhotoImage(self.mapFile.resize((round(0.75*w),round(0.75*h)), Resampling.LANCZOS))
        self.mapCanvas = Canvas(self.child, width=round(0.75 * w), height=round(0.75 * h))
        map = self.mapCanvas.create_image(0, 0, image = self.img, anchor=NW)
        self.mapCanvas.grid(column=1,columnspan=8,row=0,rowspan=6)

        # self.map = Label(self.mapFrame, image=self.img)
        # self.map.image = self.img
        # self.map.pack()
        # mapFrame.grid(column=1,columnspan=8,row=0,rowspan=6)


        self.mapCanvas.bind('<Button-1>', self.actionCheck)



    def setObject(self):
        # mb.showinfo("Задать объект исследования")
        self.nextStep = "set object"

    def setRegion(self):
        # mb.showinfo("Выделить область работы", message = "Необходимо отметить две точки на карте (левую верхнюю и правую нижнюю) и задать их координаты.")
        self.nextStep = "set region points"

    def setGatheringPoint(self):
        self.nextStep = "set gathering point"

    def actionCheck(self,event):
        radius = 7
        if self.nextStep == "set region points":
            flag = "topLeft"
            askForCoords(self.child,self.file,flag)
            # self.file.write(f"{flag}Local=[{event.x},{event.y}]\n")
            self.topLeftLocal = f"{flag}Local {event.x} {event.y}\n"
            self.mapCanvas.create_oval(event.x - radius, event.y + radius, event.x + radius, event.y - radius, fill='red')
            self.nextStep = "set second point"

        elif self.nextStep == "set second point":
            flag = "bottomRight"
            askForCoords(self.child,self.file,flag)
            # self.file.write(f"{flag}Local=[{event.x},{event.y}]\n")
            self.bottomRightLocal = f"{flag}Local {event.x} {event.y}\n"
            self.mapCanvas.create_oval(event.x - radius, event.y + radius, event.x + radius, event.y - radius, fill='blue')
            self.nextStep = ""

        elif self.nextStep == "set object":
            flag = "objCenter"
            self.logfile.write(f"{flag} {event.x} {event.y}\n")
            self.nextStep = "set radius"


        elif self.nextStep == "set radius":
            flag = "objRadius"
            self.logfile.close()
            self.logfile = open('log.txt', 'r',encoding="UTF-8")
            line = self.logfile.readlines()[-1]
            self.logfile.close()
            centerX = int(line.split(" ")[1])
            centerY = int(line.split(" ")[2])
            radius = round(math.sqrt((event.x - centerX)**2 + (event.y-centerY)**2))

            self.logfile = open('log.txt', 'a', encoding="UTF-8")
            self.logfile.write(f"{flag} {event.x} {event.y}\n")
            self.mapCanvas.create_oval(centerX - radius, centerY + radius, centerX + radius, centerY - radius, width=3, outline='yellow')
            self.objects.append([centerX, centerY, radius])
            self.nextStep = ""

        elif self.nextStep == "set gathering point":
            flag = "gatheringPointLocal"
            # self.logfile.write(f"{flag}=[{event.x},{event.y}]\n")
            self.gatheringPointLocal = f"{flag} {event.x} {event.y}\n"
            self.mapCanvas.create_oval(event.x - radius, event.y + radius, event.x + radius, event.y - radius, fill='yellow')
            self.nextStep = ""



    def saveData(self):
        if len(self.objects) > 0:
            # self.file.write(f"objects={self.objects}")
            # self.file.close()
            if self.choice.get == 1:
                self.shapeType = self.shape.get()

            self.file.write(f"shape {self.shapeType}\n")
            self.file.write(self.topLeftLocal)
            self.file.write(self.bottomRightLocal)
            self.file.write(self.gatheringPointLocal)
            self.file.write(f"objects {len(self.objects)}\n")
            for j in range(len(self.objects)):
                self.file.writelines(str(self.objects[j]) + "\n")

            self.file.close()
            self.child.destroy()
        else:
            mb.showinfo("Задание","Список заданий пуст!")




