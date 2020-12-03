import pandas as pd
import numpy as np
from numpy import format_float_scientific as flt
from pygimli import meshtools as mt

nx = 401
nz = 81


def num_converter(num):
    string = flt(abs(num), exp_digits=4, precision=14, trim='k', unique=False)
    string = string.replace('e', 'E')
    return string


data = pd.read_csv('model_0.csv')

x = np.unique(data['#X'])
z = np.unique(data['Z'])[::-1]
R = np.array(data['Rho'])

x_c = [num_converter(i) for i in x]
z_c = [num_converter(i) for i in z]
Rho_c = [num_converter(i) for i in R]

Rho = np.zeros(shape=(nx, nz))

for i in range(nx):
    Rho[i] = R[nz*i:nz*(i+1)]

Rho = np.transpose(Rho)

for i in range(nz):
    R[nx*i:nx*(i+1)] = Rho[i]

Rho = R

grid = mt.createGrid(x=np.linspace(0, 1000, nx), y=np.linspace(0, -200, nz))
res = np.array(mt.nodeDataToCellData(grid, Rho))
res = res.reshape(nx - 1, nz - 1)

with open('out_model.mod2d', 'w') as f:
    print('software: ZondRES2d', file=f)
    print('units: m', file=f)
    print('topoc:  {}'.format(num_converter(0)), file=f)
    print('x_grid:', file=f)

    f.write(' ')
    [f.write(i + '  ') for i in x_c]
    f.write('\n')

    print('z_grid:', file=f)

    f.write(' ')
    [f.write(i + '  ') for i in z_c]
    f.write('\n')

    print('topo_grid:', file=f)

    f.write(' ')
    [f.write(num_converter(0) + '  ') for i in x_c]
    f.write('\n')

    print('topo_shift:  {}'.format(num_converter(0)), file=f)

    f.write('cutting:  0 0 ')
    [f.write('{}'.format(i) + ' 0 ') for i in x]
    f.write('{}'.format(x[-1]) + ' 0 ')
    [f.write('{}'.format(i) + ' -200 ') for i in x[::-1]]
    f.write('0 0 0 0 0')
    f.write('\n')

    print('paramnumb: 1', file=f)
    print('paramname 1 : rho', file=f)
    print('islog 1 : no', file=f)
    print('minmaxCS 1 :  1.00000000000000E+0001  1.00000000000000E+0004', file=f)
    print('paramvalues 1 : ', file=f)

    for i in range(len(z)):
        f.write(' ')
        [f.write(num_converter(j) + '  ') for j in res[i][:]]
        f.write('\n')

    print('parammin 1 :', file=f)

    for i in range(len(z)):
        f.write(' ')
        [f.write(num_converter(0) + '  ') for j in res[i][:]]
        f.write('\n')

    print('parammax 1 :', file=f)

    for i in range(len(z)):
        f.write(' ')
        [f.write(num_converter(0) + '  ') for j in res[i][:]]
        f.write('\n')

    print('paramfix 1 :', file=f)

    for i in range(len(z)):
        [f.write('0 ') for j in res[i][:]]
        f.write('\n')
