import pygimli as pg
import pybert as pb
import numpy as np
import matplotlib.pyplot as plt
from pygimli import meshtools as mt
from pygimli.physics import ert

l = 5                          # ДЛИНА КОСЫ
L = 10                         # ДЛИНА ПРОФИЛЯ
h = 0.1                        # ШАГ М/У ЭЛЕКТРОДАМИ
N_l = int(l/h + 1)             # КОЛ-ВО ЭЛЕКТРОДОВ В КОСЕ
N_L = int(L/h + 1)             # КОЛ-ВО ЭЛЕКТРОДОВ НА ПРОФИЛЕ
x0 = 0                         # НАЧАЛО ПРОФИЛЯ ИЗМЕРЕНИЙ (НЕ ДОЛЖЕН ВЫЛЕЗАТЬ ЗА ПРЕДЕЛЫ МОДЕЛИ!)

world = mt.createWorld(start=[-5,0],end=[15,-10],layers=[-1.5,-3.2])

#ДЕЛАЕМ КРИВУЮ ДЛЯ CREATE_POLYGON
x = np.linspace(-5,15,201)
y = -2 - np.exp(-(x-6.5)**2)
curve = np.transpose(np.vstack((x,y)))

# НАТЫКАЛИ ВСЯКОГО НЕОДНОРОДНОГО
line1 = mt.createPolygon(curve,isClosed=False)
line2 = mt.createLine(start=[-5,-0.2],end=[15,-1.3])
circle = mt.createCircle(pos=[4,-0.4],radius=[0.5,0.2],marker=4)
block = mt.createRectangle(start=[6,-0.1],end=[7.5,-0.5],marker=5)

# СОЗДАЛИ МОДЕЛЬ И ПОКАЗАЛИ ЕЕ
model = world + line1 + line2 + circle + block
pg.show(model)
print(type(model))

# СХЕМА ДИПОЛЬ-ДИПОЛЬ, расст. м/у эл. = 100 м
scheme = ert.createERTData(elecs=np.linspace(start=x0,stop=x0+L,num=N_L),schemeName='dd')

# СДЕЛАЛИ ДИСКРЕТИЗАЦИЮ БУДУЩЕЙ СЕТКИ КАК 10% РАССТОЯНИЯ М/У ЭЛЕКТРОДАМИ, ДЛЯ НОРМ КАЧЕСТВА
for p in scheme.sensors():
    model.createNode(p)
    model.createNode(p - [0, 0.1])

# ЗАПИЛИЛИ СЕТКУ:
mesh = mt.createMesh(model, quality=34)
# ДАТА ДЛЯ СЕТКИ:
rhomap = [[0,75],[1,50],[2,125],[3,200],[4,200],[5,300]]
# ВОТ КАК ОНА ВЫГЛЯДИТ:
pg.show(mesh,data=rhomap,label=pg.unit('res'),showMesh=True)

mesh.save('mesh')

# МОДЕЛИРУЕМ ДАННЫЕ ПРЯМОЙ ЗАДАЧИ, С ШУМОМ
data = ert.simulate('mesh.bms',scheme=scheme,res=rhomap,noiseLevel=1,noiseAbs=1e-6,seed=1337)

# ЭТО ПОКА ХЗ ЗАЧЕМ
pg.warning(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
pg.info('Simulated data', data)
pg.info('The data contains:', data.dataMap().keys())

pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)

# ФИЛЬТРУЕМ УЧАСТКИ С rho < 0:
data.remove(data['rhoa'] < 0)
pg.info('Filtered rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
data.save('origin_data.dat')
data.save('required_data.dat')

# ПОКАЗЫВАЕМ РЕЗУЛЬТАТ ПРЯМОЙ ЗАДАЧИ ДО ОБРАБОТКИ
ert.show(data)

# ЗАГРУЖАЕМ КОПИЮ ТАБЛИЧКИ
data1 = pg.load('required_data.dat') 

# ПРОЧЕСЫВАЕМ ТАБЛИЧКУ .DAT, ЧТОБЫ ОБРЕЗАТЬ ТРЕУГОЛЬНИК ДО ТРАПЕЦИИ. 
# РАССТОЯНИЕ МЕЖДУ 'a' И 'n' ЭЛЕКТРОДАМИ НЕ ДОЛЖНО ПРЕВЫШАТЬ ДЛИНУ КОСЫ (N)

for i in range(len(data['valid'])):
    if data1['n'][i] - data['a'][i] > N_l-1:
        data1['valid'][i] = 0

# РЕЗУЛЬТАТ ОБРАБОТКИ:
ert.show(data1)