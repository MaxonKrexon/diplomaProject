import numpy as np
import sympy
from sympy.abc import a, b
from sympy import sin, cos, solve
from matplotlib import pyplot as plt
import tkinter.messagebox as mb

import smoothing
from myFuncs import *


def start(base=None, objects=None, velocity=1, offset = 0, tolerance = 0.0045):
    ### Блок сортировки окружностей по расположению их центров ###
    # Сортировка производится сначала по значению координаты x, потом по
    # значению координаты y

    for i in range(np.size(objects, 0)):
        for j in range(np.size(objects, 0) - 1):
            if objects[j][0] > objects[j + 1][0]:
                bag = np.copy(objects[j + 1])
                objects[j + 1] = objects[j]
                objects[j] = bag

            if objects[j][0] == objects[j + 1][0] and objects[j][1] > objects[j + 1][1]:
                pocket = np.copy(objects[j + 1])
                objects[j + 1] = objects[j]
                objects[j] = pocket

    x = np.zeros((np.size(objects, 0), ))
    y = np.zeros((np.size(objects, 0), ))
    R = np.zeros((np.size(objects, 0), ))
    V = velocity

    for i in np.arange(0, np.size(objects, 0)):
        x[i] = objects[i][0]
        y[i] = objects[i][1]
        R[i] = objects[i][2] - offset

    qq = np.array([x, y, R], np.float32)
    # print(qq)

    ### Блок построения пути ###
    # Определение координат конечной точки
    Ln_back = np.array([[base[0] - qq[0][-1]], [base[1] - qq[1][-1]]]) / np.sqrt(
        (base[0] - qq[0][-1]) ** 2 + (base[1] - qq[1][-1]) ** 2)

    # Смещённая на 3 радиуса последней окружности конечная точка перед возвращением
    coords = np.array([float(qq[0][-1] + 3 * qq[2][-1] * Ln_back[0]), float(qq[1][-1] + 3 * qq[2][-1] * Ln_back[1])],
                      np.float32)

    # Получение середин расстояний между окружностями
    coordsMiddle = np.zeros((np.size(qq, 1) - 1, 2))

    for j in range(np.size(qq, 1) - 1):
        coordsMiddle[j][0] = (float(qq[0][j] + qq[0][j + 1]) / 2)
        coordsMiddle[j][1] = (float(qq[1][j] + qq[1][j + 1]) / 2)

    keyPoints = np.vstack((base, coordsMiddle, coords))

    print('\nКлючевые точки =\n', keyPoints)

    # Первая половина серединных касательных
    # xfh = x first half; yfh = y first half

    # Скалярное произведение радиуса, содержащего точку выхода и отрезка, содержащего центр срединного расстояния
    def eq1(xfh, yfh, xm, ym):
        eq1 = (xfh - xm) * (xfh - xj) + (yfh - ym) * (yfh - yj)
        return eq1

    # Радиус, как отрезок, содержащий точку выхода
    def eq2(xfh, yfh, Rj):
        eq2 = ((xfh - xj) ** 2 + (yfh - yj) ** 2) ** (1 / 2) - Rj
        return eq2

    roots = np.zeros((np.size(qq, 1), 200000, 2))  # <- массив, где будут храниться корни уравнения
    # формат хранения: (окружность, 200000 возможных корней, [x, y])

    # После проведения расчётов может быть получено до 200 корней, многие из которых отличаются друг от друга на 0.001.
    # Чтобы их отсеять реализована следующая процедура:
    # Перебираются все элемента массива, и если разница между соседними элементами меньше 0.1, то их
    # необходимо собрать в одну ячейку, после чего поделить на общее количество и будет получен усреднённый корень.
    # Если корни отличаются более чем на 0.1, тогда это корень выходной точки и он определяется в другую ячейку.

    rootsAvr = np.zeros((np.size(qq, 1), 4))  # <- массив, где будут храниться усреднённые корни (x1,y1,x2,y2)
    xfh = np.zeros((np.size(qq, 1), 1))
    yfh = np.zeros((np.size(qq, 1), 1))
    tolerance = 0.0045  # <- погрешность указана в аргументах функции
    precision = 0.0015  # <- точность расчёта

    # Первая половина серединных касательных (выход из окружности)
    for j in range(np.size(qq, 1)):
        i = 0  # <- вспомогательный счётчик для массива корней

        # Переменные, необходимые для расчёта уравнений
        xj = float(qq[0][j])  # <- координата Х для центра окружности
        yj = float(qq[1][j])  # <- координата Y для центра окружности
        Rj = float(qq[2][j])  # <- значение радиуса окружности
        xm = float(keyPoints[j + 1][0])  # <- координата Х для центра между окружностями
        ym = float(keyPoints[j + 1][1])  # <- координата Y для центра между окружностями

        for x in np.arange(xj - Rj, xj + Rj, precision):
            for y in np.arange(yj - Rj, yj + Rj, precision):
                result1 = eq1(x, y, xm, ym)
                result2 = eq2(x, y, Rj)

                if tolerance >= abs(result1) and tolerance >= abs(result2):
                    roots[j][i] = [x, y]
                    i += 1

        exit1 = 0  # <- счётчик корней для выхода из окружности x1 и y1
        exit2 = 0  # <- счётчик корней для выхода из окружности x2 и y2


        for m in np.arange(i - 1):
            if roots[j][m][0] < min(roots[j][:, :-1][0]) + 0.1:
                rootsAvr[j][0] += roots[j][m][0]
                rootsAvr[j][1] += roots[j][m][1]
                exit1 += 1
            else:
                rootsAvr[j][2] += roots[j][m][0]
                rootsAvr[j][3] += roots[j][m][1]
                exit2 += 1

        if exit1 == 0 or exit2 == 0:
            text = f"Расчёт неверен (выход)! Увеличьте погрешность! Текущая погрешность = {tolerance}"
            mb.showerror("Ошибка расчёта!", text)

        x1 = rootsAvr[j][0] / exit1  # <- усреднённые х1 для выхода из окружности
        y1 = rootsAvr[j][1] / exit1  # <- усреднённые y1 для выхода из окружности
        x2 = rootsAvr[j][2] / exit2  # <- усреднённые x2 для выхода из окружности
        y2 = rootsAvr[j][3] / exit2  # <- усреднённые y2 для выхода из окружности

        if j == 0:
            xfh = np.vstack((x1, x2))
            yfh = np.vstack((y1, y2))
        else:
            xfh = np.vstack((xfh, x1, x2))
            yfh = np.vstack((yfh, y1, y2))


    ### Вторая половина серединных касательных (вход в окружность) ###
    # xsh = x second half; ysh = y second half
    xsh = np.zeros((np.size(qq, 1), 1))
    ysh = np.zeros((np.size(qq, 1), 1))
    roots = np.zeros((np.size(qq, 1), 200000, 2))
    rootsAvr = np.zeros((np.size(qq, 1), 4))

    # Вторая половина серединных касательных (расчёт)
    for j in range(np.size(qq, 1)):
        i = 0  # <- вспомогательный счётчик для массива корней

        # Переменные, необходимые для расчёта уравнений
        xj = float(qq[0][j])  # <- координата Х для центра окружности
        yj = float(qq[1][j])  # <- координата Y для центра окружности
        Rj = float(qq[2][j])  # <- значение радиуса окружности
        xm = float(keyPoints[j][0])  # <- координата Х для центра между окружностями (единственное отличие от пред.)
        ym = float(keyPoints[j][1])  # <- координата Y для центра между окружностями (единственное отличие от пред.)

        for x in np.arange(xj - Rj, xj + Rj, precision):
            for y in np.arange(yj - Rj, yj + Rj, precision):
                result1 = eq1(x, y, xm, ym)
                result2 = eq2(x, y, Rj)

                if tolerance >= abs(result1) and tolerance >= abs(result2):
                    roots[j][i] = [x, y]
                    i += 1

        exit1 = 0  # <- счётчик корней для выхода из окружности x1 и y1
        exit2 = 0  # <- счётчик корней для выхода из окружности x2 и y2

        for m in np.arange(i - 1):
            if roots[j][m][0] < min(roots[j][:, :-1][0]) + 0.1:
                rootsAvr[j][0] += roots[j][m][0]
                rootsAvr[j][1] += roots[j][m][1]
                exit1 += 1
            else:
                rootsAvr[j][2] += roots[j][m][0]
                rootsAvr[j][3] += roots[j][m][1]
                exit2 += 1

        if exit1 == 0 or exit2 == 0:
            text = f"Расчёт неверен (вход)! Увеличьте погрешность! Текущая погрешность = {tolerance}"
            mb.showerror("Ошибка расчёта!", text)

        x1 = rootsAvr[j][0] / exit1  # <- усреднённые х1 для входа в окружность
        y1 = rootsAvr[j][1] / exit1  # <- усреднённые y1 для входа в окружность
        x2 = rootsAvr[j][2] / exit2  # <- усреднённые x2 для входа в окружность
        y2 = rootsAvr[j][3] / exit2  # <- усреднённые y2 для входа в окружность

        if j == 0:
            xsh = np.vstack((x1, x2))
            ysh = np.vstack((y1, y2))
        else:
            xsh = np.vstack((xsh, x1, x2))
            ysh = np.vstack((ysh, y1, y2))

        # Конец цикла

    ### Осуществление соответствия точек выхода точкам входа ###

    ind = 0

    Xpv_pre1 = np.zeros((np.size(qq, 1), 1), np.float32)
    Xpvh_pre1 = np.zeros((np.size(qq, 1), 1), np.float32)
    Ypvh_pre1 = np.zeros((np.size(qq, 1), 1), np.float32)
    Ypv_pre1 = np.zeros((np.size(qq, 1), 1), np.float32)

    Xpv_pre2 = np.zeros((np.size(qq, 1), 1), np.float32)
    Xpvh_pre2 = np.zeros((np.size(qq, 1), 1), np.float32)
    Ypvh_pre2 = np.zeros((np.size(qq, 1), 1), np.float32)
    Ypv_pre2 = np.zeros((np.size(qq, 1), 1), np.float32)

    for j in range(np.size(qq, 1)):

        ### Поиск точки выхода для первой точки входа ###
        # # Для расчёта переменной Rot_sign используется понятие псевдоскаляра.
        # # По сути, мы рассчитываем третью, псевдосоставляющую (координату)
        # # двумерной точки. Знак этой координаты указывает на направление
        # # наименьшего поворота первого вектора до совпадения со вторым

        xm = keyPoints[j][0]  # <- координата Х для центра между окружностями
        ym = keyPoints[j][1]  # <- координата Y для центра между окружностями

        rotationSign1 = (xsh[ind] - qq[0][j]) * (ysh[ind] - ym) - (xsh[ind] - xm) * (ysh[ind] - qq[1][j])

        # print("\nrotationSign1 = ", rotationSign1)

        # # В зависимости от направления поворота (а значит от расположения векторов относительно друг друга)
        # # вычисляется вектор скорости

        vel1 = None
        A = np.array([qq[0][j] - xfh[ind], qq[1][j] - yfh[ind], 0.0], np.float32)
        B = np.array([0.0, 0.0, 1.0], np.float32)

        # print("\nКосое =", ())

        if rotationSign1 > 0:
            vel1 = np.cross(A, B)
        elif rotationSign1 < 0:
            vel1 = -1.0 * np.cross(A, B)

        vel1 = np.reshape(vel1, (3, 1))

        # # Разность между векторами скорости и выхода позволяет однозначно
        # # определить нужную точку выхода для гладкого движения


        xmNext = keyPoints[j + 1][0]
        ymNext = keyPoints[j + 1][1]

        # <- Вектор выхода из окружности в середину
        vecOut1 = np.array([[xmNext - xfh[ind]], [ymNext - yfh[ind]]], np.float32)
        # <- Нормированный вектор выхода
        vecOutNorm1 = np.divide(vecOut1, np.sqrt(vecOut1[0] ** 2 + vecOut1[1] ** 2))
        # <- Нормированный вектор скорости
        velNorm1 = np.divide(vel1, np.sqrt(vel1[0] ** 2 + vel1[1] ** 2 + vel1[2] ** 2))
        # <- Разность вектора скорости и вектора выхода
        vecDifference1 = np.array(vecOutNorm1 - [[velNorm1[0]], [velNorm1[1]]], np.float32)
        

        if float(np.sqrt(vecDifference1[0] ** 2 + vecDifference1[1] ** 2)) < 0.001:
            Xpv_pre1[j] = xfh[ind]
            Ypv_pre1[j] = yfh[ind]
            Xpvh_pre1[j] = xsh[ind]
            Ypvh_pre1[j] = ysh[ind]
        else:
            Xpv_pre1[j] = xfh[ind + 1]
            Ypv_pre1[j] = yfh[ind + 1]
            Xpvh_pre1[j] = xsh[ind]
            Ypvh_pre1[j] = ysh[ind]

        ### Поиск точки выхода для второй точки входа

        rotationSign2 = (xsh[ind + 1] - qq[0][j]) * (ysh[ind + 1] - ym) - (xsh[ind + 1] - xm) * (
                ysh[ind + 1] - qq[1][j])

        vel2 = None
        A = np.array([qq[0][j] - xfh[ind], qq[1][j] - yfh[ind], 0.0], np.float32)
        B = np.array([0.0, 0.0, 1.0], np.float32)
        if rotationSign2 > 0:
            vel2 = np.cross(A, B)
        elif rotationSign2 < 0:
            vel2 = -1.0 * np.cross(A, B)
        vel2 = np.reshape(vel2, (3, 1))

        # Вектор выхода из окружности в середину
        vecOut2 = np.array([[xmNext - xfh[ind]], [ymNext - yfh[ind]]], np.float32)
        # Нормированный вектор выхода
        vecOutNorm2 = np.divide(vecOut2, np.sqrt(vecOut2[0] ** 2 + vecOut2[1] ** 2))
        # Нормированный вектор скорости
        velNorm2 = np.divide(vel2, np.sqrt(vel2[0] ** 2 + vel2[1] ** 2 + vel2[2] ** 2))
        # Разность вектора скорости и вектора выхода
        vecDifference2 = np.array(vecOutNorm2 - [[velNorm2[0]], [velNorm2[1]]], np.float32)
        

        if float(np.sqrt(vecDifference2[0] ** 2 + vecDifference2[1] ** 2)) < 0.001:
            Xpv_pre2[j] = xfh[ind]
            Ypv_pre2[j] = yfh[ind]
            Xpvh_pre2[j] = xsh[ind + 1]
            Ypvh_pre2[j] = ysh[ind + 1]
        else:
            Xpv_pre2[j] = xfh[ind + 1]
            Ypv_pre2[j] = yfh[ind + 1]
            Xpvh_pre2[j] = xsh[ind + 1]
            Ypvh_pre2[j] = ysh[ind + 1]

        ind += 2

    pointIn1 = np.array([Xpvh_pre1, Ypvh_pre1], np.float32)
    pointOut1 = np.array([Xpv_pre1, Ypv_pre1], np.float32)
    pointIn2 = np.array([Xpvh_pre2, Ypvh_pre2], np.float32)
    pointOut2 = np.array([Xpv_pre2, Ypv_pre2], np.float32)


    ind = 0
    pi = 3.14159265
    xIn = np.zeros((np.size(qq, 1),))
    yIn = np.zeros((np.size(qq, 1),))
    xOut = np.zeros((np.size(qq, 1),))
    yOut = np.zeros((np.size(qq, 1),))
    arcL = np.zeros((np.size(qq, 1),))
    Ts = np.zeros((np.size(qq, 1),))

    for j in range(np.size(qq, 1)):

        # # На основе одинакового или различного поворота до совпадения радиусов,
        # # содержащих точки входа и выхода и входных радиуса и вектора
        # # определяется угол дуги (внутренний или развёрнутый угол)

        # Вектор входа в окружность
        vect11 = np.array([pointIn1[0][j] - keyPoints[j][0], pointIn1[1][j] - keyPoints[j][1]])
        vect21 = np.array([pointOut1[0][j] - qq[0][j], pointOut1[1][j] - qq[1][j]])  # Радиус, содержащий точку выхода
        vect31 = np.array([pointIn1[0][j] - qq[0][j], pointIn1[1][j] - qq[1][j]])  # Радиус, содержащий точку входа
        # Псевдоскаляр вектора входа и радиуса, содержащего точку входа
        curvProd11 = vect11[0] * vect31[1] - vect31[0] * vect11[1]
        # Псевдоскаляр радиусов содержащих точку входа и точку выхода
        curvProd21 = vect31[0] * vect21[1] - vect21[0] * vect31[1]


        if np.sign(curvProd11) == np.sign(curvProd21):
            fis1 = 2 * pi - np.arccos(((pointOut1[0][j] - qq[0][j]) * (pointIn1[0][j] - qq[0][j]) + (
                    pointOut1[1][j] - qq[1][j]) * (pointIn1[1][j] - qq[1][j])) / (np.sqrt(
                (pointOut1[0][j] - qq[0][j]) ** 2 + (pointOut1[1][j] - qq[1][j]) ** 2) * np.sqrt(
                (pointIn1[0][j] - qq[0][j]) ** 2 + (pointIn1[1][j] - qq[1][j]) ** 2)))
        else:
            fis1 = np.arccos(((pointOut1[0][j] - qq[0][j]) * (pointIn1[0][j] - qq[0][j]) + (
                    pointOut1[1][j] - qq[1][j]) * (pointIn1[1][j] - qq[1][j])) / (np.sqrt(
                (pointOut1[0][j] - qq[0][j]) ** 2 + (pointOut1[1][j] - qq[1][j]) ** 2) * np.sqrt(
                (pointIn1[0][j] - qq[0][j]) ** 2 + (pointIn1[1][j] - qq[1][j]) ** 2)))

        vect12 = np.array([pointIn2[0][j] - keyPoints[j][0], pointIn2[1][j] - pointOut2[1][j]])
        vect22 = np.array([pointOut2[0][j] - qq[0][j], pointOut2[1][j] - qq[1][j]])
        vect32 = np.array([pointIn2[0][j] - qq[0][j], pointIn2[1][j] - qq[1][j]])
        curvProd12 = vect12[0] * vect32[1] - vect32[0] * vect12[1]
        curvProd22 = vect32[0] * vect22[1] - vect22[0] * vect32[1]


        if np.sign(curvProd12) == np.sign(curvProd22):
            fis2 = 2 * pi - np.arccos(((pointOut2[0][j] - qq[0][j]) * (pointIn2[0][j] - qq[0][j]) + (
                    pointOut2[1][j] - qq[1][j]) * (pointIn2[1][j] - qq[1][j])) / (np.sqrt(
                (pointOut2[0][j] - qq[0][j]) ** 2 + (pointOut2[1][j] - qq[1][j]) ** 2) * np.sqrt(
                (pointIn2[0][j] - qq[0][j]) ** 2 + (pointIn2[1][j] - qq[1][j]) ** 2)))
        else:
            fis2 = np.arccos(((pointOut2[0][j] - qq[0][j]) * (pointIn2[0][j] - qq[0][j]) + (
                    pointOut2[1][j] - qq[1][j]) * (pointIn2[1][j] - qq[1][j])) / (np.sqrt(
                (pointOut2[0][j] - qq[0][j]) ** 2 + (pointOut2[1][j] - qq[1][j]) ** 2) * np.sqrt(
                (pointIn2[0][j] - qq[0][j]) ** 2 + (pointIn2[1][j] - qq[1][j]) ** 2)))

        if fis1 < fis2:
            xIn[ind] = pointIn1[0][j]
            yIn[ind] = pointIn1[1][j]
            xOut[ind] = pointOut1[0][j]
            yOut[ind] = pointOut1[1][j]
            arcL[ind] = fis1
            Ts[ind] = (2 * pi * qq[2][j] + (fis1 * qq[2][j])) / V

        else:
            xIn[ind] = pointIn2[0][j]
            yIn[ind] = pointIn2[1][j]
            xOut[ind] = pointOut2[0][j]
            yOut[ind] = pointOut2[1][j]
            arcL[ind] = fis2
            Ts[ind] = (2 * pi * qq[2][j] + (fis2 * qq[2][j])) / V

        ind += 1


    # Вычисление времени движения по прямым
    TsL1 = np.zeros(np.size(xIn, 0))
    TsL2 = np.zeros(np.size(xIn, 0))
    for k in range(np.size(xIn, 0)):
        # # Время движения по прямым вычисляется простым делением расстояния на
        # # скорость (так как скорость постоянна)
        tsl1 = np.sqrt((keyPoints[k + 1][0] - xOut[k]) ** 2 + (keyPoints[k + 1][1] - yOut[k]) ** 2) / V
        tsl2 = np.sqrt((keyPoints[k][0] - xIn[k]) ** 2 + (keyPoints[k][1] - yIn[k]) ** 2) / V
        TsL1[k] = np.float32(tsl1)
        TsL2[k] = np.float32(tsl2)

    # Определение времени движения по сглаженному участку
    ind = 1
    Tcurv = np.zeros(np.size(qq, 1) - 1)
    for k in range(np.size(qq, 1) - 1):
        [P, vel, _] = smoothing.start(np.float32([xOut[k], yOut[k]]),
                                      np.float32([keyPoints[k + 1][0], keyPoints[k + 1][1]]),
                                      np.float32([xIn[ind], yIn[ind]]), V)
        ind += 1
        aux = np.zeros((np.size(P, 0)), complex)
        for j in range(np.size(P, 0)):
            aux[j] = complex(P[j][0], P[j][1])

        L = sum(abs(np.diff(aux)))

        t = np.zeros(np.size(P, 0), complex)
        for m in range(np.size(P, 0)):
            t[m] = aux[m] / np.float32(vel[m])

        Tcurv[k] = sum(abs(np.diff(t)))


    Ln1 = np.zeros((np.size(qq, 1), 2))
    Ln2 = np.zeros((np.size(qq, 1), 2))

    for k in range(np.size(qq, 1)):
        # # Направляющие вектора вычисляются, как частные от деления координат
        # # векторов на их длины. Определяются для установки точного наклона
        # # относительно осей координат без трудоёмкого вычисления углов этого
        # # наклона

        ln1 = np.array([keyPoints[k + 1][0] - xOut[k], keyPoints[k + 1][1] - yOut[k]]) / np.sqrt(
            (keyPoints[k + 1][0] - xOut[k]) ** 2 + (keyPoints[k + 1][1] - yOut[k]) ** 2)
        ln2 = np.array([xIn[k] - keyPoints[k][0], yIn[k] - keyPoints[k][1]]) / np.sqrt(
            (xIn[k] - keyPoints[k][0]) ** 2 + (yIn[k] - keyPoints[k][1]) ** 2)
        Ln1[k][0] = ln1[0]  ## Ln1 - для векторов выхода
        Ln1[k][1] = ln1[1]
        Ln2[k][0] = ln2[0]  ## Ln2 - для векторов входа
        Ln2[k][1] = ln2[1]

    # Определение доп углов в движении по окружностям

    w = V / qq[2][0:]

    psi1k = np.zeros(np.size(qq, 1))
    psi2k = np.zeros(np.size(qq, 1))

    for j in range(np.size(qq, 1)):

        # Переменные, необходимые для расчёта уравнений
        xj = float(qq[0][j])  # <- координата Х для центра окружности
        yj = float(qq[1][j])  # <- координата Y для центра окружности
        Rj = float(qq[2][j])  # <- значение радиуса окружности
        XIN = float(xIn[j])  # <- координата Х для центра между окружностями
        YIN = float(yIn[j])  # <- координата Y для центра между окружностями

        psi1 = np.zeros(4)
        psi2 = np.zeros(4)

        # Фазы или доп углы вычисляются из уравнений конечных состояний

        from sympy.abc import a, b

        expr1 = sin(a) * Rj + xj - XIN
        expr2 = cos(b) * Rj + yj - YIN

        ans = np.array(solve([expr1, expr2], [a, b], check=False))

        for l in range(4):
            psi1[l] = ans[l][1]
            psi2[l] = ans[l][0]

        # Данные уравнения имеют вероятность выдачи 4 корней. Поэтому из них
        # осуществляется выбор нужного нам через простейшие тождества
        if (np.size(psi1) > 1) and (np.size(psi2) > 1):
            if (
                    np.round(np.real(qq[0][j] + qq[2][j] * np.sin(w[j] * Ts[j] + psi1[0])), 2) == np.round(xOut[j]),
                    2) and (
                    np.round(np.real(qq[1][j] + qq[2][j] * np.cos(w[j] * Ts[j] + psi2[0])), 2) == np.round(yOut[j]), 2):
                psi1k[j] = psi1[0]
                psi2k[j] = psi2[0]

            elif (
                    np.round(np.real(qq[0][j] + qq[2][j] * np.sin(w[j] * Ts[j] + psi1[1])), 2) == np.round(xOut[j]),
                    2) and (
                    np.round(np.real(qq[1][j] + qq[2][j] * np.cos(w[j] * Ts[j] + psi2[1])), 2) == np.round(yOut[j]), 2):
                psi1k[j] = psi1[1]
                psi2k[j] = psi2[1]

            elif np.round(np.real(qq[0][j] + qq[2][j] * np.sin(w[j] * Ts[j] + psi1[2])), 2) == np.round(xOut[j],
                                                                                                      2) and np.round(
                np.real(qq[1][j] + qq[2][j] * np.cos(w[j] * Ts[j] + psi2[2])), 2) == np.round(yOut[j], 2):
                psi1k[j] = psi1[2]
                psi2k[j] = psi2[2]

            elif (np.round(np.real(qq[0][j] + qq[2][j] * np.sin(w[j] * Ts[j] + psi1[3])), 2) == np.round(xOut[j],
                                                                                                       2)) and np.round(
                np.real(qq[1][j] + qq[2][j] * np.cos(w[j] * Ts[j] + psi2[3])), 2) == np.round(yOut[j], 2):
                psi1k[j] = psi1[3]
                psi2k[j] = psi2[3]

        else:
            psi1k[j] = psi1[0]
            psi2k[j] = psi2[0]


    #######################################################################################
    ############------------БЛОК-ПОСТРОЕНИЯ-ТРАЕКТОРИИ-------------------##################
    #######################################################################################

    # Построение всего пути
    # Первая прямая - первая дуга - цикл сглаживание + дуга - последняя прямая
    # Построение со сглаживанием

    ind1 = 0
    ind2 = 0

    V0 = 0.1 * V
    fi0 = arcL[0]
    fi = pi - fi0

    Vf = V - ((V - V0) * fi) / pi
    a = (V - Vf) / 2
    b = (V + Vf) / 2
    dt = (qq[2][0] * fi0) / (2 * b)

    # Массивы времени и скорости для заполнения
    vv = 0
    tt = 0
    xt11 = []
    yt11 = []
    xt21 = []
    yt21 = []

    for t in np.arange(0, TsL2[0] + Ts[0], 0.01):
        if TsL2[0] >= t >= 0:
            xt11.append(keyPoints[0][0] + Ln2[0][0] * V * t)
            yt11.append(keyPoints[0][1] + Ln2[0][1] * V * t)
            vv = np.vstack((vv, V))
            tt = np.vstack((tt, t))
            ind1 += 1
        elif t >= TsL2[0]:
            Vt = a * np.cos((pi * (t - TsL2[0])) / dt) + b
            ww = Vt / qq[2][0]
            xt21.append(qq[0][0] + qq[2][0] * np.sin(ww * (t - TsL2[0]) + psi1k[0]))
            yt21.append(qq[1][0] + qq[2][0] * np.cos(ww * (t - TsL2[0]) + psi2k[0]))
            vv = np.vstack((vv, Vt))
            tt = np.vstack((tt, t))
            ind2 += 1

    vv = vv[1:]
    tt = tt[1:]
    xt11 = np.array(xt11, np.float32)
    yt11 = np.array(yt11, np.float32)
    xt21 = np.array(xt21, np.float32)
    yt21 = np.array(yt21, np.float32)

    # Цикл сглаживание - дуга
    ind3 = 0
    ind4 = 0
    time_init = np.float32(TsL2[0] + Ts[1])
    Xt = np.zeros((np.size(qq, 1) - 1, 30))
    Yt = np.zeros((np.size(qq, 1) - 1, 30))
    timeK = 0
    ww = []
    xt1 = []
    yt1 = []
    ind4 = 0

    for k in range(np.size(qq, 1) - 1):

        timeK = np.float32(time_init + Tcurv[k] + Ts[k + 1])

        [Xx, VV, TT] = smoothing.start(np.float32([xOut[k], yOut[k]]),
                                       np.float32([keyPoints[k + 1][0], keyPoints[k + 1][1]]),
                                       np.float32([xIn[k + 1], yIn[k + 1]]), V)

        for j in range(np.size(Xx, 0)):
            Xt[k][j] = Xx[j][0]
            Yt[k][j] = Xx[j][1]
            vv = np.vstack((vv, VV[j]))
            tt = np.vstack((tt, time_init + np.transpose(TT[j])))

        for t in np.arange(time_init, timeK, 0.01):

            if t >= time_init + Tcurv[k]:
                fi0 = arcL[k + 1]
                fi = pi - fi0
                Vf = V - ((V - V0) * fi) / pi
                a = (V - Vf) / 2
                b = (V + Vf) / 2
                dt = (qq[2][k + 1] * fi0) / (2 * b)
                Vt = a * np.cos((pi * (t - (time_init + Tcurv[k]))) / dt) + b
                ww.append(Vt / qq[2][k + 1])

                xt1.append(qq[0][k + 1] + qq[2][k + 1] * np.sin(ww[ind4] * (t - (time_init + Tcurv[k])) + psi1k[k + 1]))
                yt1.append(qq[1][k + 1] + qq[2][k + 1] * np.cos(ww[ind4] * (t - (time_init + Tcurv[k])) + psi2k[k + 1]))
                vv = np.vstack((vv, Vt))
                tt = np.vstack((tt, t))
                ind4 = ind4 + 1

        time_init = time_init + Tcurv[k] + Ts[k + 1]

    xt1 = np.array(xt1)
    yt1 = np.array(yt1)

    # Построение небольшого сглаженного участка к отрезку обратного пути

    ln = np.array([[base[0] - keyPoints[-1][0]], [base[1] - keyPoints[-1][1]]]) / np.sqrt(
        (base[0] - keyPoints[-1][0]) ** 2 + (base[1] - keyPoints[-1][1]) ** 2)
    P0_little_smooth = np.array([xOut[-1], yOut[-1]])
    P1_little_smooth = np.array([keyPoints[-1][0], keyPoints[-1][1]])
    # P2_little_smooth = np.array([keyPoints[-1][0] + 0.5 * ln[0], keyPoints[-1][1] + 0.5 * ln[1]])
    P2_little_smooth = np.copy(base)

    [Xx, VV, TT] = smoothing.start(np.float32(P0_little_smooth), np.float32(P1_little_smooth),
                                   np.float32(P2_little_smooth), V)
    Xt_little_smooth = np.zeros(np.size(Xx, 0))
    Yt_little_smooth = np.zeros(np.size(Xx, 0))

    for j in range(np.size(Xx, 0)):
        Xt_little_smooth[j] = Xx[j][0]
        Yt_little_smooth[j] = Xx[j][1]

    vv = np.vstack((vv, VV))
    tt = np.vstack((tt, timeK + TT))
    

    ########################################################
    ###############------БЛОК-ОТРИСОВКИ------###############
    ########################################################

    # Окружности
    for j in range(np.size(qq, 1)):
        x = qq[0][j] + qq[2][j] * np.cos(np.arange(0, 2 * pi, 0.01))
        y = qq[1][j] + qq[2][j] * np.sin(np.arange(0, 2 * pi, 0.01))
        plt.plot(x, y, 'g')
        plt.plot(qq[0][j], qq[1][j], 'og')

    # Центральные отрезки

    for j in range(np.size(qq, 1) - 1):
        x = np.array([qq[0][j], qq[0][j + 1]])
        y = np.array([qq[1][j], qq[1][j + 1]])
        plt.plot(x, y, '-c')

    # Середины центральных отрезков

    for k in range(np.size(keyPoints, 0)):
        plt.plot(keyPoints[k][0], keyPoints[k][1], 'ob')

    for k in range(np.size(qq, 1)):
        # Точки входа из окружностей
        plt.plot(xIn[k], yIn[k], 'or')

        # Точки выхода в окружности
        plt.plot(xOut[k], yOut[k], 'om')

    plt.plot(xt11, yt11, '--r') # <- первая прямая
    plt.plot(xt21, yt21, '+b') # <- обход первой окружности

    # Сглаживание - дуга
    for j in range(np.size(qq,1)-1):
        plt.plot(Xt[j], Yt[j], '--k') # <- сглаженная прямая между окружностями

    plt.plot(xt1, yt1, '+b') # <- обход всех остальных окружностей

    # % Сглаженный участок к обратной прямой
    plt.plot(Xt_little_smooth, Yt_little_smooth, '--g')


    plt.axis("scaled")
    plt.grid()

    #Раздление траекторий обхода окружностей
    circlePath = np.zeros((np.size(qq, 1),np.size(xt1)*2 ,2))
    # print(np.size(circlePath,0),np.size(circlePath,1),np.size(circlePath,2))

    for j in range(np.size(qq, 1)):
        index = 0
        if j == 0:
            for k in range(np.size(xt21)):
                circlePath[j][index][0] = xt21[k]
                circlePath[j][index][1] = yt21[k]
                index += 1
        else:
            for k in range(np.size(xt1)):
                if (qq[0][j] - qq[2][j]) <= xt1[k] <= (qq[0][j] + qq[2][j]) and (qq[1][j] - qq[2][j]) <= yt1[k] <= (qq[1][j] + qq[2][j]):
                    circlePath[j][index][0] = xt1[k]
                    circlePath[j][index][1] = yt1[k]
                    index += 1

    # Разделение траекторий между окружностями

    x = 0
    y = 1

    
    file = open('coordinates.txt','a',encoding="UTF-8")

    for j in range(np.size(xt11)):
        pass
        file.write(f"{xt11[j]} {yt11[j]}\n")


    for j in range(np.size(qq,1)):
        for k in range(np.size(circlePath, 1)):
            if circlePath[j][k][x] != 0.0 and circlePath[j][k][y] != 0.0:
                # pass
                file.write(f"{circlePath[j][k][x]} {circlePath[j][k][y]}\n")

        if j < np.size(qq, 1) - 1:
            for k in range(np.size(Xt,1)):
                pass
                file.write(f"{Xt[j][k]} {Yt[j][k]}\n") # <- сглаженная прямая между окружностями


    for m in range(np.size(Xt_little_smooth)):
        pass
        file.write(f"{Xt_little_smooth[m]} {Yt_little_smooth[m]}\n")


    file.close()


    return xIn[0], yIn[0]


