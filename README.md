# diplomaProject
This software was made as diploma project in Robotics in may-june 2022. It is interactive graphic application for automatization of area surveilance and exploring processes.
It was realized using python 3.8 with numpy, sympy and tkinter libraries.There is a situation where this software could be used.
Imagine you are a fire department worker and you need to know how big is area of forest fire. Of course it's better to know from the height and UAV's
will help you with that, but you are a single person and you can pilot only one UAV at the time. What if you had a dozen of UAV that could be group and explore area
in boundaries that you already know? That would be extreamly helpful! This software is the solution. Here you can:
1) Load your own map as image and set its real coordinates (in degrees)
2) Select areas that you need to explore
3) Group UAV's together for one task or make them do different ones simultenously
4) Calculate the route from starting point to areas and back
5) Get the information from sensors and combine it with coordinates on map

A little bit through the sctipts: <br>
addTaskWindow.py - script that creates new window with map and functionality to set real coordinates, select areas to explore <br>
addForCoords.py - script that creates new window every time you need to set read coordinates <br>
colors.py - set of colors to use in graphics etc <br>
distribution - script that distribure areas to explore between starting points by distance <br>
getData.py - script for modelling heat map of explored surface and show it in new window <br>
myFuncs.py - script with custom funcs that shoud work as their analogies in MATLAB <br>
routePlanning.py - script that calculates route for every single UAV <br>
smoothing.py - scipt that used to make route more real <br>
startMission.py - script that combine all information about areas to explore, starting positions and starts computing <br>
tk_gui.py - main window script <br>

Данное ПО было разработано в качестве дипломной работы бакалавра по направлению "Мехатроника и Робототехника" в мае-июне 2022 года. Представляет собой интерактивное графическое приложение для решения задач автоматизации процесса исследования/наблюдения местности.

Запуск программы осуществляется посредством выполнения скрипта tk_gui.py
