from matplotlib import pyplot as plt
import pygimli as pg
from pygimli import meshtools as mt
import numpy as np
import os
import shutil

#####################################################################

verbose = 0

# ПАРАМЕТРЫ ПРЯМОУГОЛЬНОЙ МОДЕЛИ (РЕГУЛИРУЕТСЯ):
x_f = 0
x_l = 580
z_f = 0

# ОТРИЦАТЕЛЬНОЕ ЗНАЧЕНИЕ ГЛУБИНЫ (!)
z_l = -60.4

# dx = (x_l - x_f)/(nx-1)
nx = 233
nz = 41

#####################################################################

# ВОТ ТАК СОЗДАЕТСЯ ТРИАНГУЛИРОВАННАЯ МОДЕЛЬ:

world = mt.createWorld(start=[x_f, z_f], end=[x_l, z_l], layers=[z_l/3, 2*z_l/3])

# НАТЫКАЛИ ВСЯКОГО НЕОДНОРОДНОГО
line = mt.createLine(start=[x_f, z_l/6], end=[x_l, z_l/4])
circle = mt.createCircle(pos=[5*x_l/7, z_l/10], radius=[x_l/50, z_l/20], marker=4)
block = mt.createRectangle(start=[4*x_l/8, z_l/12], end=[4.5*x_l/8, 2*z_l/12], marker=5)

# СОЗДАЛИ МОДЕЛЬ
model = world + line + circle + block

# ЗАПИЛИЛИ СЕТКУ:
mesh_triangular = mt.createMesh(model, quality=34)

if os.path.exists('models'):
    shutil.rmtree('models')
os.mkdir('models')
os.chdir('models')
mesh_triangular.save('mesh_0')
pg.show(mesh_triangular)

# ЗНАЧЕНИЯ СОПРОТИВЛЕНИЙ В ОБЪЕКТАХ:
rhomap = [[0, 75], [1, 50], [2, 125], [3, 200], [4, 200], [5, 300]]
map = ''
for i in range(len(rhomap)):
    map += str(i) + ' ' + str(rhomap[i][1]) + ' '

with open('map_0.txt', 'w') as res:
    print(map, file=res)
    res.close()

os.chdir('..')

# ИЗВЛЕКАЕМ ЗНАЧЕНИЯ МАРКЕРОВ В КАЖДОЙ ЯЧЕЙКЕ, ЗАТЕМ ВРУЧНУЮ ЗАПОЛНЯЕМ СЕТКУ СОПРОТИВЛЕНИЯМИ
markers = []
triangular_res = []
for i in mesh_triangular.cells():
    markers.append(i.marker())

for i in markers:
    triangular_res.append(rhomap[i][1])

# ЗНАЧЕНИЯ В ЯЧЕЙКАХ ТРЕУГОЛЬНОЙ СЕТКИ:
triangular_res = np.array(triangular_res)

#####################################################################

# ЗАДАЧА - СОЗДАТЬ ТУ ЖЕ СЕТКУ В ПРЯМОУГОЛЬНОМ ВИДЕ, ЗАПИСАТЬ ЗНАЧЕНИЯ В УЗЛАХ В .DAT-ТАБЛИЧКУ
X = np.linspace(x_f, x_l, nx)
Z = np.linspace(z_f, z_l, nz)
mesh_rectangular = mt.createGrid(x=X, y=Z)

# ФОРМА ПРЯМОУГОЛЬНОЙ СЕТКИ
pg.show(mesh_rectangular)

# ЗНАЧЕНИЯ СОПРОТИВЛЕНИЙ В ПРЯМОУГОЛЬНЫХ ЯЧЕЙКАХ МОЖНО ПОЛУЧИТЬ С ПОМОЩЬЮ ИНТЕРПОЛЯЦИИ:
rectangular_res = np.array(mt.interpolate(mesh_rectangular, mesh_triangular, triangular_res))

# ПОКАЗЫВАЕМ ЗАПОЛНЕННУЮ МОДЕЛЬ В ВИДЕ ПРЯМОУГОЛЬНОЙ СЕТКИ
pg.show(mesh_rectangular, rectangular_res)

# ОСТАЛОСЬ ПОЛУЧИТЬ ЗНАЧЕНИЯ В УЗЛАХ И ЗАПИСАТЬ В ТАБЛИЦУ
node_rho = mt.cellDataToNodeData(mesh=mesh_rectangular, data=rectangular_res)

table_of_nodes = np.zeros(shape=(nx, nz))
for i in range(nx):
    for j in range(nz):
        table_of_nodes[i][j] = np.round(node_rho[j*nx+i])
table_of_nodes = np.transpose(table_of_nodes)

#####################################################################

if os.path.exists('test'):
    shutil.rmtree('test')
os.mkdir('test')
os.chdir('test')

# ЗАПИСЫВАЕМ ВСЕ В .DAT (Для Алексея)
with open('model_0.dat', 'w') as model:
    print("%-7s%-9s%-13s%-7s" % ('#X', 'Z', 'log_Rho', 'Rho'), file=model)
    for i in range(nx):
        for j in range(nz):
            print("%-7.2f%-9.2f%-13.2f%-7.2f" % (X[i], Z[j], np.log10(table_of_nodes[j][i]),
                                                 table_of_nodes[j][i]), file=model)
    model.close()

# ЗАПИСЬ В .СSV ДЛЯ МОЕГО АЛГОРИТМА ПРЯМОЙ ЗАДАЧИ
with open('model_0.csv', 'w') as model:
    print('#X,Z,log_Rho,Rho', file=model)
    for i in range(nx):
        for j in range(nz):
            print(str(np.round(X[i], decimals=2)) + ',' + str(np.round(Z[j], decimals=2)) + ',' + str(
                np.round(np.log10(table_of_nodes[j][i]), decimals=2)) + ',' + str(
                np.round(table_of_nodes[j][i], decimals=2)), file=model)
    model.close()

os.chdir('..')

# plt.figimage(table_of_nodes)

if verbose == 1:
    plt.show()
