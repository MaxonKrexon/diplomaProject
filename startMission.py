import numpy as np
import matplotlib.pyplot as plt

import distribution
import routePlanning
import distribution as dist
import tkinter.messagebox as mb

def startMission(bases = None):
    file = open('taskInfo.txt','r',encoding="UTF-8")
    lines = file.readlines()
    file.close()

    # Получение из файла задачи следующих данных
    # Реальные координаты левого верхнего угла карты
    topLeft = np.array([np.float32(lines[0].split(" ")[1]), np.float32(lines[0].split(" ")[2][:-1])])
    # Реальные координаты правого нижнего угла карты
    bottomRight = np.array([np.float32(lines[1].split(" ")[1]), np.float32(lines[1].split(" ")[2][:-1])])
    # Вид построения
    shape = lines[2].split(" ")[1][:-1]
    # Локальные координаты левого верхнего угла (значение пикселей на изображении)
    topLeftLocal = np.array([int(lines[3].split(" ")[1]), int(lines[3].split(" ")[2][:-1])])
    # Локальные координаты правого нижнего угла (значение пикселей на изображении)
    bottomRightLocal = np.array([int(lines[4].split(" ")[1]), int(lines[4].split(" ")[2][:-1])])
    # Локальные координаты точки сбора (значение пикселей на изображении)
    gatheringPointLocal = np.array([int(lines[5].split(" ")[1]) - topLeftLocal[0], bottomRightLocal[1] - (int(lines[5].split(" ")[2][:-1]) - topLeftLocal[1])])

    # Количество объектов для исследования
    objAmount = int(lines[6].split(" ")[1][:-1])

    objects = np.zeros((objAmount,3))

    x = 0
    y = 1
    R = 2
    # Параметры объектов из файла (в пикселях)
    for j in range(objAmount):
        objects[j][x] = int(lines[7+j].split(",")[x][1:]) - topLeftLocal[0]
        objects[j][y] = bottomRightLocal[1] - (int(lines[7+j].split(",")[y]) - topLeftLocal[1])
        objects[j][R] = int(lines[7+j].split(",")[R][:-2])


    north = topLeft[0]
    south = bottomRight[0]
    west = topLeft[1]
    east = bottomRight[1]

    x_width = abs((111.3*np.cos(np.deg2rad((north+south)/2))) * (west-east)) # km
    y_width = abs(111.3 * (north-south)) # km
    x_pixel_value = x_width/abs(topLeftLocal[0] - bottomRightLocal[0]) # km - значение 1 px в км
    y_pixel_value = y_width/abs(topLeftLocal[1] - bottomRightLocal[1]) # km

    realObjects = []
    for j in range(len(objects)):
        newX = round(x_pixel_value * objects[j][0],3)
        newY = round(y_pixel_value * objects[j][1],3)
        newR = round(np.sqrt((((objects[j][2] + objects[j][0])*x_pixel_value) - newX)**2 + (((objects[j][2] + objects[j][1])*y_pixel_value)-newY)**2), 3)
        realObjects.append([newX,newY,newR])


    realObjects = np.array(realObjects)
    realGatheringPoint = np.array([gatheringPointLocal[0] * x_pixel_value, gatheringPointLocal[1] * y_pixel_value])
    realBases=np.zeros((len(bases),2))
    for j in range(len(bases)):
        realBases[j][x] = (int(bases[j][x]) - topLeftLocal[0]) * x_pixel_value
        realBases[j][y] = (bottomRightLocal[1] - (int(bases[j][y]) - topLeftLocal[1])) * y_pixel_value


    if shape == "Single" and len(bases) == 1:
        file = open('coordinates.txt','w', encoding="UTF-8")
        file.close()
        routePlanning.start(realBases[0],realObjects, 0.122, tolerance=0.0095)

    if shape == "Single" and len(bases) > 1:
        distribution.start(realBases, realObjects)

    mb.showinfo("Уведомление","Расчёт окончен!")
