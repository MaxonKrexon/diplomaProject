import math
import pygame
import os
import colors
import tkinter.messagebox as mb
import tkinter.simpledialog as sd


def button(screen, position, text):
    font = pygame.font.SysFont("Arial", 36)
    text_render = font.render(text, True, (255, 0, 0))
    x, y, w, h = text_render.get_rect()
    x, y = position
    pygame.draw.line(screen, (150, 150, 150), (x, y), (x + w, y), 5)  # <- Нижняя граница кнопки
    pygame.draw.line(screen, (150, 150, 150), (x, y - 2), (x, y + h), 5)  # <- Левая граница кнопки
    pygame.draw.line(screen, (50, 50, 50), (x, y + h), (x + w, y + h), 5)  # <- Верхняя граница кнопки
    pygame.draw.line(screen, (50, 50, 50), (x + w, y + h), [x + w, y], 5)  # <- Правая граница кнопки
    pygame.draw.rect(screen, (100, 100, 100), (x, y, w, h))  # <- Заливка
    return screen.blit(text_render, (x, y))


x = 0
y = 1
pygame.init()
screen = pygame.display.set_mode((1024, 600))
clock = pygame.time.Clock()
map = pygame.image.load(os.path.join('map', 'FIRMS_24hrs[@103.3,54.7,12z].jpg'))
exitButton = button(map, (900, 500), "Выход")
saveTaskButton = None
resetButton = None
checkbox = ''
objects = []

while True:
    screen.blit(map, (0, 0))
    pygame.display.update()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print(objects)
            pygame.quit()

        if pygame.mouse.get_pressed(3)[0]:  # <- Если нажали ЛКМ
            center = pygame.mouse.get_pos()  # <- Получить позицию курсора
            if checkbox == 'select region':  # <- Если нажатие было по кнопке добавления задач
                checkbox = 'second point'  # <- Задать значение новой задачи для контейнера проверки
                saveTaskButton = button(map, (350, 500), "Сохранить")
                resetButton = button(map, (520, 500), "Сброс")
                pygame.draw.circle(map, (255, 0, 0), center, 5)
                topPoint = sd.askstring("Введите координаты", "Координаты точки в формате: 45.6789; 98.7654")
                print(topPoint)


            elif saveTaskButton.collidepoint(pygame.mouse.get_pos()):  # <- Если было нажатие по кнопке Сохранить
                checkbox = 'none'
                pygame.quit()
                mb.showinfo("Уведомление", "Сохранено!")
                print(objects)

            elif resetButton.collidepoint(pygame.mouse.get_pos()):
                map = pygame.image.load(os.path.join('map', 'FIRMS_24hrs[@103.3,54.7,12z].jpg'))
                screen.blit(map, (0, 0))
                exitButton = button(map, (900, 500), "Выход")
                objects = []
                checkbox = 'none'

            elif exitButton.collidepoint(pygame.mouse.get_pos()):
                pygame.quit()

            # if addPointButton.collidepoint(pygame.mouse.get_pos()):
            #     addMeetPointButton = button(map, (650, 300),"Точка сбора")
            #     addBasePointButton = button(map, (650, 400),"Точка базы")
            #     checkbox = 'add point'
            #
            # elif addBasePointButton.collidepoint(pygame.mouse.get_pos()):
            #     checkbox = 'add base'
            #
            # elif addMeetPointButton.collidepoint(pygame.mouse.get_pos()):
            #     checkbox = 'add meet'

        if pygame.mouse.get_pressed(3)[2] and checkbox == 'new task':  # <- Если после этого нажали ПКМ
            position = pygame.mouse.get_pos()
            radius = round(math.sqrt((position[x] - center[x]) ** 2 + (position[y] - center[y]) ** 2))
            objects.append([center[x], center[y], radius])
            pygame.draw.circle(map, (0, 255, 0), center, radius, 2)

        if pygame.mouse.get_pressed(3)[2] and checkbox == 'second point':
            position = pygame.mouse.get_pos()
            pygame.draw.circle(map, (0, 0, 255), position, 5)
            region = pygame.Rect(center[x], center[y], abs(center[x] - position[x]), abs(center[y] - position[y]))
            pygame.draw.rect(map, (0, 0, 255), region, 1)
