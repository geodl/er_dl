import pygimli as pg
import numpy as np
import matplotlib.pyplot as plt
from pygimli import meshtools as mt
from pygimli.physics import ert
import pandas as pd
import os
import shutil

# ПОДРАЗУМЕВАЕТСЯ, ЧТО ФАЙЛЫ МОДЕЛЕК ЛЕЖАТ В ПАПКЕ 'models' (!)
# ЕСЛИ ИХ ТУДА ЗАРАНЕЕ НЕ ПОЛОЖИТЬ, СКРИПТ УМРЕТ

# ИМЯ ДИРЕКТОРИИ, КУДА БУДУТ СОХРАНЯТЬСЯ ФАЙЛЫ РЕШЕНИЙ. МОЖЕТ БЫТЬ ИЗМЕНЕНО НА ПРОИЗВОЛЬНОЕ.
directory = 'solutions'

# ФУНКЦИЯ, МОДЕЛИРУЮЩАЯ ДАННЫЕ ПРЯМОЙ ЗАДАЧИ, ПРИНИМАЕТ АРГУМЕНТЫ:

# model = .DAT ТАБЛИЧКА, ФОРМАТ: 'model_{number}.dat'
# number = ПОРЯДКОВЫЙ НОМЕР МОДЕЛЬКИ, number = {1,2,..,10000+}
# verbose = 0 (1, ЧТОБЫ ПОКАЗАТЬ КАРТИНКИ)
# x_f, x_l, z_f, z_l, nx, nz = ПАРАМЕТРЫ ПРЯМОУГОЛЬНОЙ МОДЕЛИ (СМ. НИЖЕ)
# cable_length = ДЛИНА КОСЫ (М)
# elecs_step = ШАГ М/У ЭЛЕКТРОДАМИ

# ФОРМАТ ФАЙЛОВ РЕШЕНИЙ -- 'origin_data_{number}.dat'

#############################################################################################

# ПАРАМЕТРЫ ПРЯМОУГОЛЬНОЙ МОДЕЛИ:
X_f = 0
X_l = 1000
Z_f = 0
Z_l = -200
# dx = (x_l - x_f)/(nx-1)
Nx = 401
Nz = 81

# ПАРАМЕТРЫ КОСЫ
elecs_step = 15
cable_length = 500


def direct_model(model, x_f, x_l, z_f, z_l, nx, nz, elecs_step, cable_length, number, verbose):

    n_elecs_cable = int(cable_length / elecs_step + 1)
    n_elecs_profile = int((x_l - x_f) / elecs_step + 1)

    # СЧИТКА ДАННЫХ МОДЕЛИ ИЗ ТАБЛИЦЫ .CSV
    data = pd.read_csv(model, usecols=['#X', 'Z', 'log_Rho', 'Rho'])
    X = np.linspace(x_f, x_l, nx)
    Z = np.linspace(z_f, z_l, nz)
    R = np.array(data['Rho'])

    print(len(data), nx * nz)

    # КОЛДУНСТВО ДЛЯ ПЕРЕПИСЫВАНИЯ ВЕКТОРА СОПРОТИВЛЕНИЙ
    Rho = np.zeros(shape=(nx, nz))
    for i in range(nz):
        Rho[i] = R[nz*i:nz*(i+1)]
    Rho = np.transpose(Rho)

    for i in range(nx):
        R[nx*i:nx*(i+1)] = Rho[i]

    # ГОТОВО
    Rho = R

    # ГЕНЕРИМ СЕТОЧКУ ДЛЯ PYGIMLI
    grid = mt.createGrid(x=X, y=Z)

    # КОНВЕРТАЦИЯ ЗНАЧЕНИЙ В УЗЛАХ В ЗНАЧЕНИЯ ПО ПРЯМОУГОЛЬНИЧКАМ СЕТКИ
    # НА ВЫХОДЕ ПОЛУЧАЕМ НОВЫЙ ВЕКТОР СОПРОТИВЛЕНИЙ, КОТОРЫЙ ПРИГОДЕН ДЛЯ АЛГОРИТМА ПРЯМОЙ ЗАДАЧИ
    new_Rho = mt.nodeDataToCellData(grid, Rho)

    # ВЫВОД ВХОДНОЙ МОДЕЛИ
    pg.show(grid, new_Rho)

    # СХЕМА ДИПОЛЬ-ДИПОЛЬ, расст. м/у эл. = h (м)
    scheme = ert.createERTData(elecs=np.linspace(start=x_f, stop=x_l, num=n_elecs_profile), schemeName='dd')

    # МОДЕЛИРУЕМ ДАННЫЕ ПРЯМОЙ ЗАДАЧИ, С ШУМОМ
    data = ert.simulate(grid, scheme=scheme, res=new_Rho, noiseLevel=1, noiseAbs=1e-6, seed=1337)

    pg.warning(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
    pg.info('Simulated data', data)
    pg.info('The data contains:', data.dataMap().keys())

    pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
    pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)

    # ФИЛЬТРУЕМ УЧАСТКИ С rho < 0:
    data.remove(data['rhoa'] < 0)
    pg.info('Filtered rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))

    os.chdir('..')
    os.chdir(directory)
    outname = 'origin_data_' + str(number) + '.dat'

    # ПРОЧЕСЫВАЕМ ТАБЛИЧКУ .DAT, ЧТОБЫ ОБРЕЗАТЬ ТРЕУГОЛЬНИК ДО ТРАПЕЦИИ.
    # РАССТОЯНИЕ МЕЖДУ 'a' И 'n' ЭЛЕКТРОДАМИ НЕ ДОЛЖНО ПРЕВЫШАТЬ ДЛИНУ КОСЫ (N)
    for k in range(len(data['valid'])):
        if data['n'][k] - data['a'][k] > n_elecs_cable - 1:
            data['valid'][k] = 0

    data.save(outname)
    os.chdir('..')

    # ПОКАЗЫВАЕМ РЕЗУЛЬТАТ ПРЯМОЙ ЗАДАЧИ
    ert.show(data)

    if verbose:
        plt.show()

    return()


# СОЗДАНИЕ ПАПКИ, КУДА СКЛАДЫВАТЬ ТАБЛИЧКИ С ПРЯМЫМИ ЗАДАЧАМИ.
# ЕСЛИ ОНА УЖЕ СУЩЕСТВУЕТ, УДАЛЯЕМ И СОЗДАЕМ ЗАНОВО:
if os.path.exists(directory):
    shutil.rmtree(directory, ignore_errors=True)
os.mkdir(directory)

# ИНИЦИИРУЕМ ФУНКЦИЮ ДЛЯ МОДЕЛЕЙ № 9,15,16:
for i in [0]:
    os.chdir('csv_models')
    filename = 'model_' + str(i) + '.csv'
    direct_model(model=filename, cable_length=cable_length, elecs_step=elecs_step, x_f=X_f, x_l=X_l, z_f=Z_f, z_l=Z_l,
                 nx=Nx, nz=Nz, verbose=True, number=i)
