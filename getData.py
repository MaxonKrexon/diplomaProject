import numpy as np
from tkinter import *
from PIL import Image as pilImage
from PIL import ImageTk
from PIL.Image import Resampling
import matplotlib.pyplot as plt


def getData(parent, mapFile):
    root = Toplevel(parent)
    root.title("Просмотр данных")
    root.state("zoomed")
    root.grab_set()
    root.focus_set()

    image = np.zeros((mapFile.height, mapFile.width))

    file = open('coordinates.txt', 'r', encoding="UTF-8")
    cameraAngle = np.deg2rad(94)
    altitude = 0.075  # <- 150 meters
    beta = cameraAngle / 2
    visionRadius = altitude * np.tan(beta)

    x = 0
    y = 1
    file = open('coordinates.txt', 'r', encoding="UTF-8")
    coords = file.readlines()
    file.close()
    file = open('taskInfo.txt', 'r', encoding="UTF-8")
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

    x_width = abs((111.3 * np.cos(np.deg2rad((north + south) / 2))) * (west - east))  # km
    y_width = abs(111.3 * (north - south))  # km
    x_pixel_value = x_width / abs(topLeftLocal[x] - bottomRightLocal[x])  # km - значение 1 px в км
    y_pixel_value = y_width / abs(topLeftLocal[y] - bottomRightLocal[y])  # km

    # Количество объектов для исследования
    objAmount = int(setup[6].split(" ")[1][:-1])

    objects = np.zeros((objAmount, 3))

    x = 0
    y = 1
    R = 2
    # Параметры объектов из файла (в пикселях)
    for j in range(objAmount):
        objects[j][x] = int(setup[7 + j].split(",")[x][1:])
        objects[j][y] = int(setup[7 + j].split(",")[y])
        objects[j][R] = int(setup[7 + j].split(",")[R][:-2])

    path = np.zeros((2, np.size(coords)), int)
    for j in range(np.size(coords)):
        path[x][j] = topLeftLocal[0] + (float(coords[j].split(" ")[x]) / x_pixel_value)
        path[y][j] = bottomRightLocal[1] - ((float(coords[j].split(" ")[y][:-1]) / y_pixel_value) - topLeftLocal[1])

    for j in range(np.size(coords) - 1):
        pxX = path[x][j]
        pxY = path[y][j]

        y_border = round(visionRadius / y_pixel_value)
        x_border = round(visionRadius / x_pixel_value)
        flag = -1
        for k in range(objAmount):
            if objects[k][x] - objects[k][R] - x_border <= pxX <= objects[k][x] + objects[k][R] + x_border and \
                    objects[k][y] - objects[k][R] - y_border <= pxY <= objects[k][y] + objects[k][R] + y_border:
                temp = np.random.randint(65, 80)
                flag = k
                break
            else:
                temp = np.random.randint(27, 35)
                flag = -1

        for k in range(-y_border, y_border):
            for l in range(-x_border, x_border):
                if flag != -1:
                    localTemp = temp - abs(pxX - objects[flag][x])
                else:
                    localTemp = np.random.randint(27, 35)

                image[pxY + k][pxX + l] = localTemp + np.random.randint(-8, 4)

    plt.imsave("heatMap.png", image, cmap='inferno')
    heatmapFile = pilImage.open(r"heatMap.png")

    for j in range(heatmapFile.width):
        for k in range(heatmapFile.height):
            if heatmapFile.getpixel((j, k)) == (0, 0, 3, 255):
                heatmapFile.putpixel((j, k), (mapFile.getpixel((j, k))))

    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    heatmap = ImageTk.PhotoImage(heatmapFile.resize((round(0.9 * w), round(0.9 * h))), Resampling.LANCZOS)
    heatmapCanvas = Canvas(root, width=round(0.9 * w), height=round(0.9 * h))
    heatmapCanvas.create_image(0, 0, image=heatmap, anchor=NW)
    heatmapCanvas.pack()
