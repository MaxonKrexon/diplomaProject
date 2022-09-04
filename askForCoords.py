from tkinter import *

def getCoords(x, y,root,file,flag):
    # point = [round(float(x.get()),4), round(float(y.get()),4)]
    lon = round(float(x.get()),4) # Широта
    lat = round(float(y.get()),4) # Долгота
    file.write((f'{flag} {lon} {lat}\n'))
    root.destroy()

def askForCoords(parent,file,flag):
    root = Toplevel(parent)
    root.title("Укажите координаты")
    root.grab_set()
    root.focus_set()
    Label(root,text="Введите координаты точки в формате 12,3456; 65,4321").grid(column=0,row=0,columnspan=4,pady=5,padx=5)

    Label(root,text="Широта").grid(column=0,row=1,pady=5,padx=5)
    latitude = StringVar()
    Entry(root,width=15, textvariable=latitude).grid(column=1,row=1,pady=5,padx=5)

    Label(root,text="Долгота").grid(column=2,row=1,pady=5,padx=5)
    longitude = StringVar()
    Entry(root,width=15, textvariable=longitude).grid(column=3,row=1,pady=5,padx=5)

    Button(root,text="Сохранить", command=lambda: getCoords(latitude,longitude,root,file,flag)).grid(row=2,column=1,columnspan=3,padx=5,pady=5, sticky=E + W)

