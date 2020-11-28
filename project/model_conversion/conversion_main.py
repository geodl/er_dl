from matplotlib import pyplot as plt
import pygimli as pg
from pygimli import meshtools as mt
import numpy as np
import os
import shutil
from project.config import common_config

# ЭТО ГЛАВНЫЙ СКРИПТ
# КОММЕНТЫ К ЭТОЙ ФУНКЦИИ В ТЕСТОВОМ СКРИПТЕ !
# x_f, x_l, z_f, z_l, nx, nz -- КОНСТАНТЫ ДЛЯ ЭТОГО ПРОЕКТА. ПРОЩЕ ЗАДАТЬ ИХ ЗАРАНЕЕ

# ПАРАМЕТРЫ ПРЯМОУГОЛЬНОЙ МОДЕЛИ:
X_f = 0
X_l = 1000
Z_f = 0
Z_l = -200
# dx = (x_l - x_f)/(nx-1)
Nx = 401
Nz = 81

# if os.path.exists('dat_models'):
#     shutil.rmtree('dat_models')
# if os.path.exists('csv_models'):
#     shutil.rmtree('csv_models')
# os.mkdir('dat_models')
# os.mkdir('csv_models')

dat_models_dir = common_config.root_dir / 'project/model_conversion/dat_models'
csv_models_dir = common_config.root_dir / 'project/model_conversion/csv_models'

dat_models_dir.mkdir(parents=True, exist_ok=True)
csv_models_dir.mkdir(parents=True, exist_ok=True)


def convert_model(model_file, rho_file, x_f, x_l, z_f, z_l, nx, nz, verbose, number):

    mesh_triangular = pg.load(model_file)
    pg.show(mesh_triangular)

    with open(rho_file, 'r') as res:
        map = res.read()
        res.close()

    map = np.fromstring(map, dtype=float, sep=' ')
    rhomap = []
    k = 0
    while k < len(map):
        rhomap.append([map[k], map[k+1]])
        k += 2
    rhomap = np.array(rhomap)

    os.chdir('..')

    markers = []
    triangular_res = []

    for i in mesh_triangular.cells():
        markers.append(i.marker())

    for i in markers:
        triangular_res.append(rhomap[i][1])

    triangular_res = np.array(triangular_res)

    X = np.linspace(x_f, x_l, nx)
    Z = np.linspace(z_f, z_l, nz)
    mesh_rectangular = mt.createGrid(x=X, y=Z)
    pg.show(mesh_rectangular)

    rectangular_res = np.array(mt.interpolate(mesh_rectangular, mesh_triangular, triangular_res))
    pg.show(mesh_rectangular, rectangular_res)

    node_rho = mt.cellDataToNodeData(mesh=mesh_rectangular, data=rectangular_res)

    table_of_nodes = np.zeros(shape=(nx, nz))
    for i in range(nx):
        for j in range(nz):
            table_of_nodes[i][j] = np.round(node_rho[j * nx + i])
    table_of_nodes = np.transpose(table_of_nodes)

    # ИМЕНА ВЫХОДНЫХ ФАЙЛОВ:
    outfile_csv = 'model_' + str(number) + '.csv'
    outfile_dat = 'model_' + str(number) + '.dat'

    os.chdir('dat_models')
    with open(outfile_dat, 'w') as model:
        print("%-7s%-9s%-13s%-7s" % ('#X', 'Z', 'log_Rho', 'Rho'), file=model)
        for i in range(nx):
            for j in range(nz):
                print("%-7.2f%-9.2f%-13.2f%-7.2f" % (X[i], Z[j], np.log10(table_of_nodes[j][i]),
                                                     table_of_nodes[j][i]), file=model)
        model.close()
    os.chdir('..')

    os.chdir('csv_models')
    with open(outfile_csv, 'w') as model:
        print('#X,Z,log_Rho,Rho', file=model)
        for i in range(nx):
            for j in range(nz):
                print(str(np.round(X[i], decimals=2)) + ',' + str(np.round(Z[j], decimals=2)) + ',' + str(
                    np.round(np.log10(table_of_nodes[j][i]), decimals=2)) + ',' + str(
                    np.round(table_of_nodes[j][i], decimals=2)), file=model)
    model.close()
    os.chdir('..')

    if verbose:
        plt.show()


for i in [1]:
    os.chdir('models')
    filename = 'mesh_' + str(i) + '.bms'
    rhoname = 'map_' + str(i) + '.txt'
    convert_model(model_file=filename, rho_file=rhoname, x_f=X_f, x_l=X_l,
                  z_f=Z_f, z_l=Z_l, nx=Nx, nz=Nz, verbose=True, number=i)
