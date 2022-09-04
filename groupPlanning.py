import numpy as np
from matplotlib import pyplot as plt
import routePlanning

def groupPlanning(base = None, objects = None, shape = None):

    cameraAngle = np.deg2rad(94)
    altitude = 0.075  # <- 75 meters
    beta = cameraAngle / 2
    visionRadius = altitude * np.tan(beta)
    interval = 1.9 * visionRadius
    quantity = 4  # <- Количество БПЛА в группе

    # objects = np.array([[11, 15, 0.3],
    #                     [3, 12, 0.3],
    #                     [1, 2, 0.3]])
    #
    # base = np.array([5, 5])  # <- место сбора БПЛА
    x = 0
    y = 1

    #### Поворот строя ####

    xIn, yIn = routePlanning.start(base, objects, 0.022)
    plt.plot(base[x] + visionRadius * np.cos(np.arange(0, 2 * np.pi, 0.01)),
             base[y] + visionRadius * np.sin(np.arange(0, 2 * np.pi, 0.01)))
    plt.close()
    A = np.array([base[x], base[y]])
    B = np.array([base[x], yIn])
    C = np.array([xIn, yIn])

    a = np.sqrt((B[x] - C[x]) ** 2 + (B[y] - C[y]) ** 2)
    b = np.sqrt((A[x] - C[x]) ** 2 + (A[y] - C[y]) ** 2)
    c = np.sqrt((A[x] - B[x]) ** 2 + (A[y] - B[y]) ** 2)

    alpha = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    beta = np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
    gamma = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

    print("\nalpha = ", np.rad2deg(alpha))
    print("\nbeta = ", np.rad2deg(beta))
    print("\ngamma = ", np.rad2deg(gamma))

    if yIn > base[y] and xIn < base[x]:
        direction = 1
    else:
        direction = -1

    from scipy.spatial.transform import Rotation as rot

    rotMatrix = np.array([[np.cos(alpha), -np.sin(alpha) * direction, 0],
                          [np.sin(alpha) * direction, np.cos(alpha), 0],
                          [0, 0, 1]])
    q = rot.from_matrix(rotMatrix)

    ##############################################

    if shape == "Квадрат":
        formationBases = np.zeros((int(np.sqrt(quantity)), int(np.sqrt(quantity)), 2))
        if np.sqrt(quantity) % 1 == 0:  # <- проверка на правильность построения
            if quantity % 2 == 0:  # <- рассмастривается случай, когда в построении чётное количество БПЛА
                xLeftFront = base[x] - (interval * ((np.sqrt(quantity) / 2) - 1) + interval / 2)
                yLeftFront = base[y]

            else:
                xLeftFront = base[x] - (interval * (np.sqrt(quantity) // 2))
                yLeftFront = base[y]

            for j in np.arange(int(np.sqrt(quantity))):
                for k in np.arange(int(np.sqrt(quantity))):
                    if k == j == 0:
                        formationBases[j][k][x] = xLeftFront
                        formationBases[j][k][y] = yLeftFront
                    else:
                        formationBases[j][k][x] = xLeftFront + j * interval
                        formationBases[j][k][y] = yLeftFront - k * interval

            radiusOffset = np.zeros(int(np.sqrt(quantity)))

            # xfirst, yfirst = routePlanning.start(base, objects, 0.022)
            # plt.show()

            if xIn > xLeftFront:
                offsetDirection = 1
            else:
                offsetDirection = -1

            for j in range(int(np.sqrt(quantity))):
                radiusOffset[j] = offsetDirection * (formationBases[j][0][x] - base[x])

            ind = 0
            for j in np.arange(int(np.sqrt(quantity))):
                for k in np.arange(int(np.sqrt(quantity))):
                    vector = np.array([((formationBases[0][0][x] + j * interval) - formationBases[0][0][x]),
                                       ((formationBases[0][0][y] - k * interval) - formationBases[0][0][y]), 0])
                    formationBases[j][k] = formationBases[0][0] + q.apply(vector)[0:2]
                    plt.plot(formationBases[j][k][x], formationBases[j][k][y],'ob')
                    # plt.plot(formationBases[j][k][x] + visionRadius * np.cos(np.arange(0, 2 * np.pi, 0.01)),
                    #          formationBases[j][k][y] + visionRadius * np.sin(np.arange(0, 2 * np.pi, 0.01)))
                    ind += 1

            # for j in np.arange(int(np.sqrt(quantity))):
            #     for k in np.arange(int(np.sqrt(quantity))):
            #         routePlanning.start(formationBases[j][k], objects, 0.022, radiusOffset[j])

            print(q.as_quat())
        else:
            print("\nПри построении квадратом необходимо равное количество БПЛА на каждую сторону построения!")


    if shape == "Колонна":
        formationBases = np.zeros((quantity, 2))
        # Построение имеет начало в точке сбора
        # Начальный строй также не ориентирован
        for j in range(quantity):
            formationBases[j][x] = base[x]
            formationBases[j][y] = base[y] - j * interval

        for j in range(quantity):
            vector = np.array([0, ((formationBases[0][y] - j * interval) - formationBases[0][y]), 0])
            formationBases[j] = formationBases[0] + q.apply(vector)[0:2]
            plt.plot(formationBases[j][x], formationBases[j][y], 'ob')
            # plt.plot(formationBases[j][x] + visionRadius * np.cos(np.arange(0, 2 * np.pi, 0.01)),
            #          formationBases[j][y] + visionRadius * np.sin(np.arange(0, 2 * np.pi, 0.01)))

        # for j in np.arange(quantity):
        #     routePlanning.start(formationBases[j], objects, 0.022)


    if shape == "Шеренга":
        formationBases = np.zeros((quantity, 2))
        # Построение имеет начало в точке сбора
        # Начальный строй также не ориентирован
        for j in range(quantity):
            formationBases[j][x] = base[x] + j * interval
            formationBases[j][y] = base[y]

        if xIn > formationBases[0][x]:
            offsetDirection = 1
        else:
            offsetDirection = -1

        radiusOffset = np.zeros(quantity)

        for j in range(quantity):
            radiusOffset[j] = offsetDirection * (formationBases[j][x] - base[x])

        for j in range(quantity):
            vector = np.array([((formationBases[0][x] + j * interval) - formationBases[0][x]), 0, 0])
            formationBases[j] = formationBases[0] + q.apply(vector)[0:2]
            plt.plot(formationBases[j][x], formationBases[j][y], 'ob')
            # plt.plot(formationBases[j][x] + visionRadius * np.cos(np.arange(0, 2 * np.pi, 0.01)),
            #          formationBases[j][y] + visionRadius * np.sin(np.arange(0, 2 * np.pi, 0.01)))


        # for j in np.arange(quantity):
        #     routePlanning.start(formationBases[j], objects, 0.022, radiusOffset[j])


    # for j in range(quantity):
    #         plt.plot(formationBases[j][x],formationBases[j][y],'ob')

    plt.plot(base[x], base[y], 'or')
    plt.axis("scaled")
    plt.show()

#
# base = [3,2]
# objects = [[1,1,0.3],[5,8,0.25]]
# shape = "Шеренга"
#
# groupPlanning(base,objects,shape)
