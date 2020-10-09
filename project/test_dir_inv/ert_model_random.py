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


answer = rand.choice([1, 2, 3, 4, 5, 6, 7, 8])

if answer == '1':
    # Horizontal layering model
    world = mt.createWorld(start=[0, 0], end=[1000, -200], layers=gen_y_for_1_till_5())
    geom = world

elif answer == '2':
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
    world = mt.createWorld(start=[0, 0], end=[1000, -200], layers=y_coord)


    def gen_coord_for_bulge(y0):
        x0 = rand.uniform(300, 600)
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

elif answer == '3':
    # Pinch of layer
    def gen_3_y_for_pinch():
        y = []
        for i in range(1, 3):
            y.append(rand.uniform(-45, -5))
        y = sorted(y)
        if any(abs(y[i + 1] - y[i]) < 15 for i in range(len(y) - 1)):
            y = gen_3_y_for_pinch()
        else:
            pass
        return y


    y_coord = gen_3_y_for_pinch()
    world = mt.createWorld(start=[0, 0], end=[1000, -200], layers=y_coord)


    def gen_coord_for_pinch(y0):
        x0 = rand.uniform(300, 600)
        x1 = x0 + rand.uniform(4, 33)
        y1 = y0 - rand.uniform(5, 15)
        x2 = x1 + rand.uniform(4, 33)
        y2 = y0 - rand.uniform(5, 15)
        x3 = x2 + rand.uniform(4, 33)
        return [(x0, y0), (x1, y1), (x2, y2), (x3, y0), (x0, y0)]


    num_layers = rand.choice([1, 2, 3])
    poly1 = mt.createPolygon(gen_coord_for_pinch(y_coord[0]), isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)
    poly2 = mt.createPolygon(gen_coord_for_pinch(y_coord[1]), isClosed=True,
                             addNodes=3, interpolate='linear', marker=1)
    if num_layers == 1:
        poly = poly1
    elif num_layers == 2:
        poly = poly2
    else:
        poly = poly1 + poly2
    geom = world + poly

elif answer == '4':
    world = mt.createWorld(start=[0, 0], end=[1000, -200], )

    x1 = rand.uniform(0, 200)
    x2 = rand.uniform(800, 1000)
    y3 = rand.uniform(-200, -10)

    choice = rand.choice([1, 2, 3, 4])
    if choice == 1:
        poly = mt.createPolygon([(1000, -200), (x1, -200), (x2, 0), (1000, 0), (1000, -200)], isClosed=True,
                                addNodes=3, interpolate='linear', marker=2)
    elif choice == 2:
        poly = mt.createPolygon([(0, 0), (x1, 0), (x2, -200), (0, -200), (0, 0)], isClosed=True,
                                addNodes=3, interpolate='linear', marker=2)
    elif choice == 3:
        x3 = rand.uniform(700, 800)
        poly1 = mt.createPolygon([(x3, 0), (0, y3), (0, 0), (x3, 0)], isClosed=True,
                                 addNodes=3, interpolate='linear', marker=2)
        poly2 = mt.createPolygon([(1000, -200), (x1, -200), (x2, 0), (1000, 0), (1000, -200)], isClosed=True,
                                 addNodes=3, interpolate='linear', marker=3)
        poly = poly1 + poly2
    elif choice == 4:
        x3 = rand.uniform(200, 300)
        poly1 = mt.createPolygon([(x3, 0), (1000, y3), (1000, 0), (x3, 0)], isClosed=True,
                                 addNodes=3, interpolate='linear', marker=2)
        poly2 = mt.createPolygon([(0, 0), (x1, 0), (x2, -200), (0, -200), (0, 0)], isClosed=True,
                                 addNodes=3, interpolate='linear', marker=3)
        poly = poly1 + poly2
    else:
        pass
    geom = world + poly

elif answer == '5':
    # Integration
    world = mt.createWorld(start=[0, 0], end=[1000, -200])

    # Creating coordinates of the implementation
    x0 = rand.uniform(300, 700)
    y0 = rand.uniform(0, -60)


    def width():
        y1 = rand.uniform(-200, -5)
        y2 = rand.uniform(-200, -5)
        y = [y1, y2]
        if abs(y1 - y2) < 30:
            y = width()
        else:
            pass
        return y


    y = width()
    poly1 = mt.createPolygon([(x0, y0), (0, y[0]), (0, y[1]), (x0, y0)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)
    poly2 = mt.createPolygon([(x0, y0), (1000, y[0]), (1000, y[1]), (x0, y0)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)

    num_integration = rand.choice([1, 2, 3])
    if num_integration == 1:
        poly = poly1
    elif num_integration == 2:
        poly = poly2
    else:
        poly = poly1 + poly2
    geom = world + poly

elif answer == '6':
    # Lens

    y0 = rand.uniform(-45, -5)


    def gen_coord_for_lens(y0):
        x0 = rand.uniform(300, 600)
        x1 = x0 + rand.uniform(4, 33)
        y1 = y0 - rand.uniform(5, 20)
        x2 = x1 + rand.uniform(4, 33)
        y2 = y0 - rand.uniform(5, 20)
        x3 = x2 + rand.uniform(4, 33)
        return [(x0, y0), (x1, y1), (x2, y2), (x3, y0), (x0, y0)]


    def gen_y_coord_for_lens():
        y = []
        for i in range(1, 3):
            y.append(rand.uniform(-50, -15))
        y = sorted(y)
        if any(abs(y[i + 1] - y[i]) < 15 for i in range(len(y) - 1)):
            y = gen_y_coord_for_lens()
        else:
            pass
        return y


    y_coord = gen_y_coord_for_lens()

    world = mt.createWorld(start=[0, 0], end=[1000, -200], layers=y_coord)
    poly = mt.createPolygon(gen_coord_for_lens(y0), isClosed=True,
                            addNodes=3, interpolate='linear', marker=(len(y_coord) + 2))
    geom = world + poly

elif answer == '7':
    # Layer contact

    # Create coordinates of the contacting layers
    width = rand.uniform(10, 40)


    def wid():
        y_low = rand.uniform(-60, -20)
        if y_low + width > -5:
            y_low = wid()
        else:
            pass
        return y_low


    y_low = wid()
    x_low = rand.uniform(415, 585)
    y_up = y_low + width
    x_up = x_low + rand.uniform(-115, 115)

    world = mt.createWorld(start=[0, 0], end=[1000, -200], layers=[y_low])
    # Creating contact layers
    poly1 = mt.createPolygon([(0, y_low), (x_low, y_low), (x_up, y_up), (0, y_up), (0, y_low)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=3)
    poly2 = mt.createPolygon([(1000, y_low), (x_low, y_low), (x_up, y_up), (1000, y_up), (1000, y_low)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=4)
    # Creating a layer that overlaps the contacts
    poly3 = mt.createPolygon([(0, 0), (1000, 0), (1000, y_up), (0, y_up), (0, 0)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=2)
    geom = world + poly1 + poly2 + poly3

elif answer == '8':
    # Fault

    # Create coordinates of the fault
    y0 = rand.uniform(-10, -50)
    x0 = rand.uniform(415, 585)
    x1 = x0 + rand.uniform(-115, 115)
    length_of_fault = rand.uniform(20, 75)
    x00 = x0 + length_of_fault
    x11 = x1 + length_of_fault

    world = mt.createWorld(start=[0, 0], end=[1000, -200], layers=[y0])
    # Create the body of the fault covering and adjacent layers
    poly2 = mt.createPolygon([(x0, y0), (x00, y0), (x11, -199), (x1, -199), (x0, y0)], isClosed=True,
                             addNodes=3, interpolate='linear', marker=3)
    geom = world + poly2

else:
    exit()

r = []
for i in range(1, 7):
    r1 = rand.randint(10, 1000)
    r2 = rand.randint(10, 1000)
    r.append(sorted([r1, r2]))

scheme = ert.createERTData(elecs=np.linspace(start=0, stop=10, num=41), schemeName='slm')

for p in scheme.sensors():
    geom.createNode(p)
    geom.createNode(p - [0, 0.1])

mesh = mt.createMesh(geom, quality=34)

r = []
for i in range(7):
    r1 = i
    r2 = rand.randint(10, 10000)
    r.append([r1, r2])
print(r)

print(type(mesh))
pg.show(mesh, data=r, label=pg.unit('res'), showMesh=True)
plt.show()
# esh.save('mesh_test')
