import pygimli as pg
import pybert as pb
import numpy as np
import matplotlib.pyplot as plt
from pygimli import meshtools as mt
from pygimli.physics import ert
import random as rand
import math


# the function outputs a list of random y coordinates for the layer boundary (1-5 layers)
def gen_y_for_1_till_5():
    y = []
    for i in range(1, rand.randint(2, 5)):
        y.append(rand.uniform(-55, -5))
    y = sorted(y)
    if any(abs(y[i + 1] - y[i]) < 5 for i in range(len(y) - 1)):
        y = gen_y_for_1_till_5()
    else:
        pass
    return y


print('Do you want to choose a model?\n([y], n)?\n')
answer1 = input()
if answer1 == 'n':
    answer2 = rand.choice([1, 2, 3, 4, 5, 6, 7, 8])
else:
    print('To select a model, tap its number:\n'
          '[1] - Horizontal layering\n'
          '[2] - Bulge of layer\n'
          '[3] - Pinch of layer\n'
          '[4] - Sloping layering\n'
          '[5] - Integration\n'
          '[6] - Lens\n'
          '[7] - Layer contact\n'
          '[8] - Fault\n')
    answer2 = input()
if answer2 == '1':
    # Horizontal layering model
    world = mt.createWorld(start=[0, 0], end=[1000, -60], layers=gen_y_for_1_till_5())
    geom = world

elif answer2 == '2':
    # Bulge of layer
    def gen_3_y_for_bulge():
        y = []
        for i in range(1, 3):
            y.append(rand.uniform(-50, -15))
        y = sorted(y)
        if any(abs(y[i + 1] - y[i]) < 15 for i in range(len(y) - 1)):
            y = gen_3_y_for_bulge()
        else:
            pass
        return y


    y_coord = gen_3_y_for_bulge()
    world = mt.createWorld(start=[0, 0], end=[1000, -60], layers=y_coord)


    def gen_coord_for_bulge(y0):
        x0 = rand.uniform(5, 900)
        x1 = x0 + rand.uniform(4, 33)
        y1 = y0 + rand.uniform(5, 15)
        x2 = x1 + rand.uniform(4, 33)
        y2 = y0 + rand.uniform(5, 15)
        x3 = x2 + rand.uniform(4, 33)
        return [(x0, y0), (x1, y1), (x2, y2), (x3, y0), (x0, y0)]


    num_layers = rand.randrange(1, 4)
    if num_layers == 1:
        poly1 = mt.createPolygon(gen_coord_for_bulge(y_coord[0]), isClosed=True,
                                 addNodes=3, interpolate='linear', marker=3)
        poly = poly1
    elif num_layers == 2:
        poly2 = mt.createPolygon(gen_coord_for_bulge(y_coord[1]), isClosed=True,
                                 addNodes=3, interpolate='linear', marker=2)
        poly = poly2
    else:
        poly1 = mt.createPolygon(gen_coord_for_bulge(y_coord[0]), isClosed=True,
                                 addNodes=3, interpolate='linear', marker=3)
        poly2 = mt.createPolygon(gen_coord_for_bulge(y_coord[1]), isClosed=True,
                                 addNodes=3, interpolate='linear', marker=2)
        poly = poly1 + poly2
    geom = world + poly

elif answer2 == '3':
    # Pinch of layer
    def gen_3_y_for_pinch():
        y = []
        for i in range(1, 3):
            y.append(rand.uniform(-45, -5))
        y = sorted(y)
        if any(abs(y[i + 1] - y[i]) < 15 for i in range(len(y) - 1)):
            y = gen_y_for_1_till_5()
        else:
            pass
        return y


    y_coord = gen_3_y_for_pinch()
    world = mt.createWorld(start=[0, 0], end=[1000, -60], layers=y_coord)


    def gen_coord_for_pinch(y0):
        x0 = rand.uniform(5, 900)
        x1 = x0 + rand.uniform(4, 33)
        y1 = y0 - rand.uniform(5, 15)
        x2 = x1 + rand.uniform(4, 33)
        y2 = y0 - rand.uniform(5, 15)
        x3 = x2 + rand.uniform(4, 33)
        return [(x0, y0), (x1, y1), (x2, y2), (x3, y0), (x0, y0)]


    num_layers = rand.choice([1, 2, 3])
    poly1 = mt.createPolygon(gen_coord_for_pinch(y_coord[0]), isClosed=True,
                             addNodes=3, interpolate='linear', marker=1)
    poly2 = mt.createPolygon(gen_coord_for_pinch(y_coord[1]), isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)
    if num_layers == 1:
        poly = poly1
    elif num_layers == 2:
        poly = poly2
    else:
        poly = poly1 + poly2
    geom = world + poly

elif answer2 == '4':
    # Sloping layering
    print('do not ready')

elif answer2 == '5':
    # Integration
    world = mt.createWorld(start=[0, 0], end=[1000, -60], )

    # Creating coordinates of the implementation
    x0 = rand.uniform(100, 900)
    y0 = rand.uniform(-10, -50)
    width = rand.uniform(5, 40)
    y1 = y0 + width / 2
    y2 = y0 - width / 2

    poly1 = mt.createPolygon([(x0, y0), (0, y1), (0, y2), (x0, y0)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)
    poly2 = mt.createPolygon([(x0, y0), (1000, y1), (1000, y2), (x0, y0)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)

    num_integration = rand.choice([1, 2, 3])
    if num_integration == 1:
        poly = poly1
    elif num_integration == 2:
        poly = poly2
    else:
        poly = poly1 + poly2
    geom = world + poly

elif answer2 == '6':
    # Lens

    def gen_coord_for_lens_and_check_position():
        y0 = rand.uniform(-45, -5)

        def gen_coord_for_lens(y0):
            x0 = rand.uniform(5, 1000)
            x1 = x0 + rand.uniform(4, 33)
            y1 = y0 - rand.uniform(5, 20)
            x2 = x1 + rand.uniform(4, 33)
            y2 = y0 - rand.uniform(5, 20)
            x3 = x2 + rand.uniform(4, 33)
            return [(x0, y0), (x1, y1), (x2, y2), (x3, y0), (x0, y0)]

        coord_for_lens = gen_coord_for_lens(y0)
        coord_for_layers = gen_y_for_1_till_5()
        if 0 < y0 < coord_for_layers[1] or any(
                coord_for_layers[i] < y0 < coord_for_layers[i + 1] for i in range(len(coord_for_layers) - 1)) or \
                coord_for_layers[len(coord_for_layers) - 1] < y0 < -60:
            pass
        else:
            [coord_for_layers, coord_for_lens] = gen_coord_for_lens_and_check_position()
        return [coord_for_layers, coord_for_lens]


    coord = gen_coord_for_lens_and_check_position()
    world = mt.createWorld(start=[0, 0], end=[1000, -60], layers=coord[0])
    poly = mt.createPolygon(coord[1], isClosed=True,
                            addNodes=3, interpolate='linear', marker=(len(coord[0]) + 2))
    geom = world + poly

elif answer2 == '7':
    # Layer contact

    # Create coordinates of the contacting layers
    width = rand.uniform(10, 40)
    y_low = rand.uniform(-50, -20)
    x_low = rand.uniform(115, 885)
    y_up = y_low + width
    x_up = x_low + rand.uniform(-115, 115)

    world = mt.createWorld(start=[0, 0], end=[1000, -60], layers=[y_low])
    # Creating contact layers
    poly1 = mt.createPolygon([(0, y_low), (x_low, y_low), (x_up, y_up), (0, y_up), (0, y_low)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=3)
    poly2 = mt.createPolygon([(1000, y_low), (x_low, y_low), (x_up, y_up), (1000, y_up), (1000, y_low)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=4)
    # Creating a layer that overlaps the contacts
    poly3 = mt.createPolygon([(0, 0), (1000, 0), (1000, y_up), (0, y_up), (0, 0)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=5)
    geom = world + poly1 + poly2 + poly3

elif answer2 == '8':
    # Fault

    # Create coordinates of the fault
    y0 = rand.uniform(-10, -40)
    y1 = -60
    x0 = rand.uniform(115, 885)
    x1 = x0 + rand.uniform(-115, 115)
    length_of_fault = rand.uniform(20, 75)
    x00 = x0 + length_of_fault
    x11 = x1 + length_of_fault

    world = mt.createWorld(start=[0, 0], end=[1000, -60])
    # Create the body of the fault covering and adjacent layers
    poly1 = mt.createPolygon([(0, -60), (0, y0), (1000, y0), (1000, -60), (0, -60)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=1)
    poly2 = mt.createPolygon([(x0, y0), (x00, y0), (x11, -60), (x1, -60), (x0, y0)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)
    poly3 = mt.createPolygon([(0, 0), (1000, 0), (1000, y0), (0, y0), (0, 0)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=3)
    geom = world + poly1 + poly2 + poly3

else:
    exit()

r = []
for i in range(1, 7):
    r1 = rand.randint(10, 1000)
    r2 = rand.randint(10, 1000)
    r.append(sorted([r1, r2]))
# СХЕМА ДИПОЛЬ-ДИПОЛЬ, расст. м/у эл. = 100 м
scheme = ert.createERTData(elecs=np.linspace(start=0, stop=10, num=41), schemeName='slm')

# СДЕЛАЛИ ДИСКРЕТИЗАЦИЮ БУДУЩЕЙ СЕТКИ КАК 10% РАССТОЯНИЯ М/У ЭЛЕКТРОДАМИ, ДЛЯ НОРМ КАЧЕСТВА
for p in scheme.sensors():
    geom.createNode(p)
    geom.createNode(p - [0, 0.1])

# ЗАПИЛИЛИ СЕТКУ:
mesh = mt.createMesh(geom, quality=34)
# ДАТА ДЛЯ СЕТКИ:
r = []
for i in range(7):
    r1 = i
    r2 = rand.randint(10, 1000)
    r.append(sorted([r1, r2]))
# ВОТ КАК ОНА ВЫГЛЯДИТ:
# pg.show(mesh,data=rhomap,label=pg.unit('res'),showMesh=True)
print(type(mesh))
pg.show(mesh, data=r, label=pg.unit('res'), showMesh=True)
plt.show()
# esh.save('mesh_test')
