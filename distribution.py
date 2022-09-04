import numpy as np
from matplotlib import pyplot as plt
import routePlanning



def distanceCalculation(startPoint=None, objects=None):
    # Далее формируется массив, в котором сохранятся дистанции от начальных точек до всех окружностей
    DTO = np.zeros((np.size(objects, 0), np.size(startPoint, 0)), np.float32)  # <- Distance to Object
    point = 0
    for n in range(np.size(startPoint, 0)):
        line = []
        for num in range(np.size(objects, 0)):
            distance = np.sqrt(
                ((startPoint[n][0] - objects[num][0]) ** 2) + ((startPoint[n][1] - objects[num][1]) ** 2))
            line.append(distance)

        for ind in range(np.size(objects, 0)):
            DTO[ind][point] = line[ind]
        line.clear()
        point += 1

    return DTO


def show(array):
    for i in range(np.size(array, 0)):
        print("\nPoint " + str(i + 1))
        for j in range(np.size(array, 1)):
            print(array[i][j])


def draw(startPoint=None, objects=None, colour = None):

    # for j in range(np.size(startPoint, 0)):
    plt.plot(startPoint[0], startPoint[1], 'ok')

    for obj in range(np.size(objects, 0)):
        x = objects[obj][0]
        y = objects[obj][1]
        R = objects[obj][2]
        plt.plot(x + R * np.cos(np.arange(0, 2 * 3.14159265, 0.01)), y + R * np.sin(np.arange(0, 2 * 3.14159265, 0.01)),
                 'k')
        # plt.plot([startPoint[0], objects[obj][0]], [startPoint[1], objects[obj][1]], f'-{colour}')

    plt.grid()
    plt.axis("scaled")
    # plt.show()


# Необходимо реализовать следующий метод
# Для каждой цели вычисляется минимальное расстояние
# от точки вылета, после чего выдаётся номер точки вылета
# А затем формируется список точек для исследования для
# каждой точки вылета.
# Например, исходя из текущих параметров, есть три точки вылета
# (столбцы 0, 1, 2)
# Сравниваем расстояния
#  |5.0 |8.06|22.84| <- Минимум 5 (1 старт)
#  |12.0|15.6|24.02| <- Минимум 12 (1 старт)
#  |20.2|19.2|18.02| <- Минимум 18 (3 старт)
#  и т.д.
# Затем формируется список задач, то есть расчёт траектории
# Зная начальную точку и координаты объектов исследования.
# Выглядит это примерно так:
# Формируется массив 3х8 (3 старта и 8 точек)
# В столбцы записываются только те координаты,
# которые следует исследовать из конкретной начальной точки
#  |3,4 |None|None|
#  |0,12|None|None|
#  |None|None|7,19|
# и т.д.
# Затем для выполнения задачи в функцию расчёта траекторий
# посылается нужный столбец массива с координатами объектов
# для конкретной стартовой точки. Надо только сделать на входе
# в эту функцию обработку случая, когда координаты имеют вид None,
# Но для этого есть команды по фильтрации нулевых элементов массива.


def sortObjects(DTO=None, startPoint=None, objects=None):
    start = np.zeros(np.size(DTO, 0), np.float32)

    for W in range(np.size(DTO, 0)):
        localMin = DTO[W][:].min()
        # 1) получено значение 5
        # необходимо найти в каком столбце было
        start[W] = np.argwhere(DTO[W] == localMin)
        # 2) получено значение 0
        # что соответствует 1-му старту = верно

    # print(start)
    # создаётся массив задач
    tasks = np.zeros((np.size(startPoint, 0), np.size(objects, 0), 3))

    for W in range(np.size(DTO, 0)):
        point = int(start[W])
        tasks[point][W] = objects[W]
        # print(objects[W])
    # print(tasks[:][:])

    return tasks


def start(startPoint, objects):

    file = open('coordinates.txt','w', encoding="UTF-8")
    file.close()
    # Координаты начальных точек
    # startPoint = np.array([[0, 0],
    #                       [10, 0],
    #                       [24, 13]])
    #
    # # Координаты центров окружностей их радиусы
    # objects = np.array([[3,4, 0.3],
    #                     [0,12, 0.4],
    #                     [7,19, 0.3],
    #                     [11,15, 0.5],
    #                     [10,6, 0.32],
    #                     [17,19, 0.3],
    #                     [19,8, 0.3],
    #                     [17,2, 0.4]])

    DTO = distanceCalculation(startPoint, objects)
    tasks = sortObjects(DTO, startPoint, objects)
    # show(tasks)
    # draw(startPoint, objects)


    # Нулевые значения, которые остались после сортировки задач между исполнителями
    # Заменяются на значения 99_999, выбранное для максимальной заметности ненужного элемента
    for point in range(np.size(startPoint, 0)):
        for obj in range(np.size(objects, 0)):
            if tasks[point][obj][0] == tasks[point][obj][1] == tasks[point][obj][2] == 0:
                tasks[point][obj][0] = 99_999
                tasks[point][obj][1] = 99_999
                tasks[point][obj][2] = 99_999

    for launch in range(np.size(startPoint, 0)):
        print('\nЗапуск из', startPoint[launch])
        print('\nДанные о целях: x/ y/ R \n')

        active = 0  # <- Счётчик текущих целей для вылета
        for i in range(np.size(objects, 0)):
            if tasks[launch][i][1] != 99_999:  # <- Проверка на лишнее значение
                active += 1

        objectsForLaunch = np.zeros((active, 3))  # <- Создание массива с данными о целях без лишних данных

        j = 0
        for i in range(np.size(objects, 0)):
            if tasks[launch][i][1] != 99_999:
                objectsForLaunch[j] = tasks[launch][i]
                j += 1

        print(objectsForLaunch)

        base = startPoint[launch]

        # colours = ['k', 'k', 'k']
        # draw(startPoint[launch], objectsForLaunch, colours[launch])

        routePlanning.start(base, objectsForLaunch, 0.122, tolerance=0.0095)
