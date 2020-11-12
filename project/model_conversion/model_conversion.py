from matplotlib import pyplot as plt
import pygimli as pg
from pygimli import meshtools as mt
import numpy as np

#####################################################################

# ПАРАМЕТРЫ ПРЯМОУГОЛЬНОЙ МОДЕЛИ (РЕГУЛИРУЕТСЯ):
x_f = 0
x_l = 580
z_f = 0

# ОТРИЦАТЕЛЬНОЕ ЗНАЧЕНИЕ ГЛУБИНЫ (!)
z_l = -60

nx = 581
nz = 161

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
pg.show(mesh_triangular)

# ЗНАЧЕНИЯ СОПРОТИВЛЕНИЙ В ОБЪЕКТАХ:
rhomap = [[0, 75], [1, 50], [2, 125], [3, 200], [4, 200], [5, 300]]

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
print(max(rectangular_res))

# ПОКАЗЫВАЕМ ЗАПОЛНЕННУЮ МОДЕЛЬ В ВИДЕ ПРЯМОУГОЛЬНОЙ СЕТКИ
pg.show(mesh_rectangular, rectangular_res)

plt.show()
