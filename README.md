<h1> diplomaProject</h1>
This software was made as diploma project in Robotics in may-june 2022. It is interactive graphic application for automatization of area surveillance and exploring processes.
It was implemented using <b>python 3.8</b> with <b>numpy, sympy and tkinter</b> libraries.There is a situation where this software could be used.<br>
Imagine you are a fire department worker and you need to know how big is area of forest fire. Of course it's better to know from the height and UAV's
will help you with that, but you are a single person and you can pilot only one UAV at the time. What if you had a dozen of UAV that could be group and explore area in boundaries that you already know? That would be extreamly helpful! This software is the solution. Here you can:
1) Load your own map as image and set its real coordinates (in degrees)
2) Select areas that you need to explore
3) Group UAV's together for one task or make them do different ones simultenously
4) Calculate the route from starting point to areas and back
5) Get the information from sensors and combine it with coordinates on map

<h4>A little bit through the sctipts: </h4><br>
<b>addTaskWindow.py</b> - script that creates new window with map and functionality to set real coordinates, select areas to explore <br>
<b>addForCoords.py</b> - script that creates new window every time you need to set real coordinates <br>
<b>colors.py</b> - set of colors to use in graphics etc <br>
<b>distribution</b> - script that distribute areas to explore between starting points by distance <br>
<b>getData.py</b> - script for modelling heat map of explored surface and show it in the new window <br>
<b>myFuncs.py</b> - script with custom funcs that work as their analogues in MATLAB <br>
<b>routePlanning.py</b> - script that calculates route for every single UAV (original language: MATLAB) <br>
<b>smoothing.py</b> - scipt that used to make route more real <br>
<b>startMission.py</b> - script that combine all information about areas to explore, starting positions and starts computing <br>
<b>tk_gui.py</b> - main window script with list of UAVs, map, buttons and feedback (didn't implemented) <br>

<h4> Here is some screenshots:</h4>
<a href="tk_gui.py">
<img src="https://user-images.githubusercontent.com/112805583/196709910-ec4ea216-1e8e-464b-a698-1ac15d79ef0e.png"></a><br>
Picture 1. Main window <br><br>

<a href="addTaskWindow.py">
<img src="https://user-images.githubusercontent.com/112805583/196711671-aa23de41-7ac0-42f1-85da-9919372e506e.png"></a><br>
Picture 2. Create task window <br><br>

<a href="askForCoords.py">
<img src="https://user-images.githubusercontent.com/112805583/196712369-e4cb9886-6303-43ff-a980-598af0c1b42c.png"></a><br>
Picture 3. Window to set real coordinates on a picture <br><br>

<a href="addTaskWindow.py">
<img src="https://user-images.githubusercontent.com/112805583/196712846-632f7934-5457-44a8-b26c-f0331175fe5c.png"></a><br>
Picture 4. Create task window with selected areas <br><br>

<a href="tk_gui.py">
<img src="https://user-images.githubusercontent.com/112805583/196713295-f36f9097-b96f-4221-9a24-6ab6996e168d.png"></a><br>
Picture 5. Main window with calculated route for two UAVs <br><br>

<a href="getData.py">
<img src="https://user-images.githubusercontent.com/112805583/196713636-1eff7b11-0607-4748-8444-f6fa4171095b.png"></a><br>
Picture 6. Window with modelled data from UAVs <br><br>

Данное ПО было разработано в качестве дипломной работы бакалавра по направлению "Мехатроника и Робототехника" в мае-июне 2022 года. Представляет собой интерактивное графическое приложение для решения задач автоматизации процесса исследования/наблюдения местности.

Запуск программы осуществляется посредством выполнения скрипта tk_gui.py
