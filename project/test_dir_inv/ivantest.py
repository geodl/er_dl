import random as rand
# fordict1 = [
#     ['key1',a[0]],
#     ['key2',a[1]],
#     ['key3',a[2]]
# ]
# dict1 = dict(fordict1)
# print(dict1)

# a=rand.randrange(1, 4)
# print(a)

#Модель разлом

Y0 = rand.uniform(-10, -40)
Y1 = -60
X0 = rand.uniform(115,885)
X1 = X0 + rand.uniform(-115,115)
DL = rand.uniform(20, 75)
X00 = X0 + DL
X11 = X1 + DL

# Слои сбоу от разлома
world = mt.createWorld(start=[0, 0], end=[1000, -60], layers=[XLOW])
# Создаём тело разлома и покрывающий слой
poly1 = mt.createPolygon([(X0, Y0), (X00, Y0), (X11, -60), (X1, -60)], isClosed=True,
                            addNodes=3, interpolate='linear', marker=2)
poly2 = mt.createPolygon([(0, O), (1000, 0), (1000, Y0), (0, Y0)], isClosed=True,
                            addNodes=3, interpolate='linear', marker=3)
geom = world + poly1 + poly2
