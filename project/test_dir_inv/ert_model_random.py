import pygimli as pg
#import pybert as pb
import numpy as np
import matplotlib.pyplot as plt
from pygimli import meshtools as mt
from pygimli.physics import ert
import random as rand

#функция выдает список из рандомных координат y для границы слоев (1-5 слоев)
def func1():
    a=[]
    for i in range(1,rand.randint(2,5)):
        a.append(rand.uniform(-55,-5))
    a=sorted(a)
    if any(abs(a[i + 1] - a[i]) < 5 for i in range(len(a) - 1)):
        a = func1()
    else:
        pass
    if any(abs(a[i + 1] - a[i]) < 5 for i in range(len(a) - 1)):
        a = func1()
    else:
        pass
    return a



#Модель горизонтальная слоистость
world = mt.createWorld(start=[0,0],end=[1000,-60],layers=func1())
geom = world

#Модель раздув пласта (горизонтальная слоистость)
#func2 как func1 только генерит 3 слоя и мощности больше 15
def func2():
    a=[]
    for i in range(1,3):
        a.append(rand.uniform(-55,-15))
    a=sorted(a)
    if any(abs(a[i + 1] - a[i]) < 15 for i in range(len(a) - 1)):
        a = func1()
    else:
        pass
    return a
y_kord=func2()
world = mt.createWorld(start=[0,0],end=[1000,-60],layers=y_kord)
#func3 генерит координаты для раздува
def func3(Y0 ):
    X0 = rand.uniform(5, 1000)
    X1 = X0+rand.uniform(4, 33)
    Y1=Y0+rand.uniform(5, 15)
    X2=X1+rand.uniform(4, 33)
    Y2=Y0+rand.uniform(5, 15)
    X3=X2+rand.uniform(4, 33)
    return [(X0,Y0), (X1,Y1), (X2,Y2), (X3,Y0)]
kol_vo=rand.randrange(1, 4)
if kol_vo==1:
    poly1 = mt.createPolygon(func3(y_kord[0]), isClosed=True,
                             addNodes=3, interpolate='linear', marker=1)
    poly=poly1
elif kol_vo==2:
    poly2 = mt.createPolygon(func3(y_kord[1]), isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)
    poly=poly2
else:
    poly1 = mt.createPolygon(func3(y_kord[0]), isClosed=True,
                            addNodes=3, interpolate='linear', marker=1)
    poly2 = mt.createPolygon(func3(y_kord[1]), isClosed=True,
                            addNodes=3, interpolate='linear', marker=2)
    poly = poly1+poly2
geom = world + poly

#Модель пережим пласта (горизонтальная слоистость)
#func4 как func1 только генерит 3 слоя и мощности больше 15
def func4():
    a=[]
    for i in range(1,3):
        a.append(rand.uniform(-45,-5))
    a=sorted(a)
    if any(abs(a[i + 1] - a[i]) < 15 for i in range(len(a) - 1)):
        a = func1()
    else:
        pass
    return a
y_kord=func4()
world = mt.createWorld(start=[0,0],end=[1000,-60],layers=y_kord)
#func5 генерит координаты для пережима
def func5(Y0 ):
    X0 = rand.uniform(5, 1000)
    X1 = X0+rand.uniform(4, 33)
    Y1=Y0-rand.uniform(5, 15)
    X2=X1+rand.uniform(4, 33)
    Y2=Y0-rand.uniform(5, 15)
    X3=X2+rand.uniform(4, 33)
    return [(X0,Y0), (X1,Y1), (X2,Y2), (X3,Y0)]
kol_vo=rand.randrange(1, 4)
if kol_vo==1:
    poly1 = mt.createPolygon(func5(y_kord[0]), isClosed=True,
                             addNodes=3, interpolate='linear', marker=1)
    poly=poly1
elif kol_vo==2:
    poly2 = mt.createPolygon(func5(y_kord[1]), isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)
    poly=poly2
else:
    poly1 = mt.createPolygon(func5(y_kord[0]), isClosed=True,
                            addNodes=3, interpolate='linear', marker=1)
    poly2 = mt.createPolygon(func5(y_kord[1]), isClosed=True,
                            addNodes=3, interpolate='linear', marker=2)
    poly = poly1+poly2
geom = world + poly

# layer1 = mt.createPolygon([[0.0, 137], [117.5, 164], [117.5, 162], [0.0, 135]],
#                           isClosed=True, marker=1, area=1)
# layer2 = mt.createPolygon([[0.0, 126], [0.0, 135], [117.5, 162], [117.5, 153]],
#                           isClosed=True, marker=2)
# layer3 = mt.createPolygon([[0.0, 110], [0.0, 126], [117.5, 153], [117.5, 110]],
#                           isClosed=True, marker=3)
#
# slope = (164 - 137) / 117.5
#
# geom = layer1 + layer2 + layer3

#Модель линза
def func1():
    a=[]
    for i in range(1,rand.randint(2,5)):
        a.append(rand.uniform(-55,-5))
    a=sorted(a)
    if any(abs(a[i + 1] - a[i]) < 5 for i in range(len(a) - 1)):
        a = func1()
    else:
        pass
    if any(abs(a[i + 1] - a[i]) < 5 for i in range(len(a) - 1)):
        a = func1()
    else:
        pass
    return a

def func7():
    Y0 = rand.uniform(-45, -5)
    # func6 генерит координаты для линзы
    def func6(Y0):
        X0 = rand.uniform(5, 1000)
        X1 = X0 + rand.uniform(4, 33)
        Y1 = Y0 - rand.uniform(5, 20)
        X2 = X1 + rand.uniform(4, 33)
        Y2 = Y0 - rand.uniform(5, 20)
        X3 = X2 + rand.uniform(4, 33)
        return [(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y0)]
    b=func6(Y0)
    a=func1()
    if 0 < Y0 < a[1] or any(a[i] < Y0 < a[i+1] for i in range(len(a) - 1)) or a[len(a)-1] < Y0 < -60:
        pass
    else:
        [a,b] = func7()
    return [a,b]
c = func7()
world = mt.createWorld(start=[0,0],end=[1000,-60],layers=c[0])
poly = mt.createPolygon(c[1], isClosed=True,
                             addNodes=3, interpolate='linear', marker=len(c[0]+1))
geom = world + poly


#Модель контакт слоёв

mosh = rand.uniform(10,40)
YLOW = rand.uniform(-50,-20)
XLOW = rand.uniform(115,885)
YUP = YLOW + mosh
XUP = XLOW + rand.uniform(-115,115)

world = mt.createWorld(start=[0,0],end=[1000,-60],layers=[XLOW])
# Создаём контактирующие пласты
poly1 = mt.createPolygon([(O,YLOW), (XLOW,YLOW), (XUP,YUP), (0,YUP)], isClosed=True,
                            addNodes=3, interpolate='linear', marker=1)
poly2 = mt.createPolygon([(1000, YLOW), (XLOW, YLOW), (XUP, YUP), (1000, YUP)], isClosed=True,
                            addNodes=3, interpolate='linear', marker=1)
# Создаём слой перекрывающий контактирующие
poly3 = mt.createPolygon([(0, 0), (0, 1000), (1000, YLOW), (0, YLOW)], isClosed=True,
                            addNodes=3, interpolate='linear', marker=1)
geom = world + poly1 + poly2 + poly3


#Модель разлом

Y0 = rand.uniform(-10, -40)
Y1 = -60
X0 = rand.uniform(115,885)
X1 = X0 + rand.uniform(-115,115)
DL = rand.uniform(20, 75)
X00 = X0 + DL
X11 = X1 + DL

# Слои сбоу от разлома
world = mt.createWorld(start=[0, 0], end=[1000, -60], layers=[XLOW])
# Создаём тело разлома и покрывающий слой
poly1 = mt.createPolygon([(X0, Y0), (X00, Y0), (X11, -60), (X1, -60)], isClosed=True,
                            addNodes=3, interpolate='linear', marker=2)
poly2 = mt.createPolygon([(0, O), (1000, 0), (1000, Y0), (0, Y0)], isClosed=True,
                            addNodes=3, interpolate='linear', marker=3)
geom = world + poly1 + poly2
