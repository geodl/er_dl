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


def convert_model(x_f, x_l, z_f, z_l, nx, nz, verbose, root_dir, number):

    filename = 'mesh_' + str(number) + '.bms'
    rhoname = 'map_' + str(number) + '.txt'

    conv_dir = root_dir / 'project/model_conversion'

    model_file = conv_dir / f'models/{filename}'
    rho_file = str(conv_dir / f'models/{rhoname}')

    if model_file.stat().st_size > 200 * 1024:
        return

    model_file = str(model_file)

    mesh_triangular = pg.load(model_file)

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

    rectangular_res = np.array(mt.interpolate(mesh_rectangular, mesh_triangular, triangular_res))

    node_rho = mt.cellDataToNodeData(mesh=mesh_rectangular, data=rectangular_res)

    table_of_nodes = np.zeros(shape=(nx, nz))
    for i in range(nx):
        for j in range(nz):
            table_of_nodes[i][j] = np.round(node_rho[j * nx + i])
    table_of_nodes = np.transpose(table_of_nodes)

    # ИМЕНА ВЫХОДНЫХ ФАЙЛОВ:
    outfile_csv = conv_dir / 'csv_models' / ('model_' + str(number) + '.csv')
    outfile_dat = conv_dir / 'dat_models' / ('model_' + str(number) + '.dat')

    with open(outfile_dat, 'w') as model:
        print("%-7s%-9s%-13s%-7s" % ('#X', 'Z', 'log_Rho', 'Rho'), file=model)
        for i in range(nx):
            for j in range(nz):
                print("%-7.2f%-9.2f%-13.2f%-7.2f" % (X[i], Z[j], np.log10(table_of_nodes[j][i]),
                                                     table_of_nodes[j][i]), file=model)
        model.close()

    with open(outfile_csv, 'w') as model:
        print('#X,Z,log_Rho,Rho', file=model)
        for i in range(nx):
            for j in range(nz):
                print(str(np.round(X[i], decimals=2)) + ',' + str(np.round(Z[j], decimals=2)) + ',' + str(
                    np.round(np.log10(table_of_nodes[j][i]), decimals=2)) + ',' + str(
                    np.round(table_of_nodes[j][i], decimals=2)), file=model)
    model.close()

    if verbose:
        plt.show()


if __name__ == "__main__":
    # ПАРАМЕТРЫ ПРЯМОУГОЛЬНОЙ МОДЕЛИ:
    X_f = 0
    X_l = 500
    Z_f = 0
    Z_l = -200
    # dx = (x_l - x_f)/(nx-1)
    Nx = 501
    Nz = 201

    if os.path.exists('dat_models'):
        shutil.rmtree('dat_models')
    if os.path.exists('csv_models'):
        shutil.rmtree('csv_models')
    os.mkdir('dat_models')
    os.mkdir('csv_models')

    num_models = len(list((common_config.root_dir / 'project/model_conversion/models').glob('*'))) // 2
    base_args = tuple([X_f, X_l, Z_f, Z_l, Nx, Nz, False, common_config.root_dir])

    args = [list(base_args) + [idx] for idx in range(num_models)]

    from python_utils.runner import Runner

    runner = Runner('process', os.cpu_count() // 2)
    runner.run(convert_model, args)

