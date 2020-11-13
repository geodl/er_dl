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
# directory = ОПИСАНО ВЫШЕ

# ФОРМАТ ФАЙЛОВ РЕШЕНИЙ -- 'origin_data_{number}.dat'


def direct_model(model, number, verbose):
    L = 580                        # ДЛИНА ПРОФИЛЯ
    l = 150                        # ДЛИНА КОСЫ
    h = 10                         # ШАГ М/У ЭЛЕКТРОДАМИ
    N_l = int(l / h + 1)           # КОЛ-ВО ЭЛЕКТРОДОВ В КОСЕ
    N_L = int(L/h + 1)             # КОЛ-ВО ЭЛЕКТРОДОВ НА ПРОФИЛЕ
    x0 = 0                         # НАЧАЛО ПРОФИЛЯ ИЗМЕРЕНИЙ (НЕ ДОЛЖЕН ВЫЛЕЗАТЬ ЗА ПРЕДЕЛЫ МОДЕЛИ!)

    # СЧИТКА ДАННЫХ МОДЕЛИ ИЗ ТАБЛИЦЫ .CSV
    data = pd.read_csv(model, usecols=['#X', 'Z', 'log_Rho', 'Rho'])
    X = np.linspace(0, 580, 233)
    Z = np.linspace(-60.4, 0, 41)
    R = np.array(data['Rho'])

    # КОЛДУНСТВО ДЛЯ ПЕРЕПИСЫВАНИЯ ВЕКТОРА СОПРОТИВЛЕНИЙ
    Rho = np.zeros(shape=(233, 41))
    for i in range(233):
        Rho[i] = R[41*i:41*(i+1)][::-1]
    Rho = np.transpose(Rho)

    for i in range(41):
        R[233*i:233*(i+1)] = Rho[i]

    # ГОТОВО
    Rho = R

    # ГЕНЕРИМ СЕТОЧКУ ДЛЯ PYGIMLI (ПОДДЕРЖИВАЕТСЯ NUMPY)
    grid = mt.createGrid(x=X, y=Z)

    # КОНВЕРТАЦИЯ ЗНАЧЕНИЙ В УЗЛАХ В ЗНАЧЕНИЯ ПО ПРЯМОУГОЛЬНИЧКАМ СЕТКИ
    # НА ВЫХОДЕ ПОЛУЧАЕМ НОВЫЙ ВЕКТОР СОПРОТИВЛЕНИЙ, КОТОРЫЙ ПРИГОДЕН ДЛЯ АЛГОРИТМА ПРЯМОЙ ЗАДАЧИ
    new_Rho = mt.nodeDataToCellData(grid, Rho)

    # ВЫВОД ВХОДНОЙ МОДЕЛИ
    pg.show(grid, new_Rho)

    # СХЕМА ДИПОЛЬ-ДИПОЛЬ, расст. м/у эл. = h (м)
    scheme = ert.createERTData(elecs=np.linspace(start=x0, stop=x0+L, num=N_L), schemeName='dd')

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
        if data['n'][k] - data['a'][k] > N_l - 1:
            data['valid'][k] = 0

    data.save(outname)
    os.chdir('..')

    # ПОКАЗЫВАЕМ РЕЗУЛЬТАТ ПРЯМОЙ ЗАДАЧИ
    ert.show(data)

    if verbose == 1:
        plt.show()

    return()


# СОЗДАНИЕ ПАПКИ, КУДА СКЛАДЫВАТЬ ТАБЛИЧКИ С ПРЯМЫМИ ЗАДАЧАМИ.
# ЕСЛИ ОНА УЖЕ СУЩЕСТВУЕТ, УДАЛЯЕМ И СОЗДАЕМ ЗАНОВО:
shutil.rmtree(directory, ignore_errors=True)
os.mkdir(directory)

# ИНИЦИИРУЕМ ФУНКЦИЮ ДЛЯ МОДЕЛЕЙ № 9,15,16:
for i in [0]:
    os.chdir('models')
    filename = 'model_' + str(i) + '.csv'
    direct_model(model=filename, verbose=1, number=i)
    os.chdir('..')
