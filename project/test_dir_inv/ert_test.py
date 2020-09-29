import pygimli as pg
import pybert as pb
import numpy as np
import matplotlib.pyplot as plt
from pygimli import meshtools as mt
from pygimli.physics import ert

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
scheme = ert.createERTData(elecs=np.linspace(start=0,stop=10,num=21),schemeName='dd')

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

# МОДЕЛИРУЕМ ДАННЫЕ ПРЯМОЙ ЗАДАЧИ, С ШУМОМ
data = ert.simulate('mesh_test.bms',scheme=scheme,res=rhomap,noiseLevel=1,noiseAbs=1e-6,seed=1337)

# ЭТО ПОКА ХЗ ЗАЧЕМ
pg.warning(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
pg.info('Simulated data', data)
pg.info('The data contains:', data.dataMap().keys())

pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)

# ФИЛЬТРУЕМ УЧАСТКИ С rho < 0:
data.remove(data['rhoa'] < 0)
pg.info('Filtered rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
data.save('ERT_test.dat') # СОХРАНИМ ЕЖЖЕ

# ПОКАЗЫВАЕМ РЕЗУЛЬТАТ ПРЯМОЙ ЗАДАЧИ
ert.show(data)

# ЗАПУСКАЕМ КАКОЙ ТО ЕРТ МЕНЕЖЕР
mng = ert.ERTManager('ERT_test.dat')

# ИНВЕРСИРУЕМ
inv = mng.invert(lam=10,verbose=True)

# РЕЗУЛЬТАТ
mng.showResultAndFit()
meshPD = pg.Mesh(mng.paraDomain) # делаем копию инверсионных данных для графика
inv.save('invert')
pg.show(mng.paraDomain)