from typing import Union, Optional, List, Tuple, Iterable

import pygimli as pg
import pybert as pb
import numpy as np
import matplotlib.pyplot as plt
from pygimli import meshtools as mt
from pygimli.core._pygimli_ import Mesh
from pygimli.physics import ert
import random as rand
import math
from pygimli.viewer.mpl import drawMesh


ResType = Union[float, np.ndarray]
PointType = Union[Tuple[float, float], np.ndarray]

rand.seed(1)
np.random.seed(1)


class ModelConfig:
    class ResValues:
        min_res = 10
        max_res = 10000

    class World(ResValues):
        left = 0
        top = 0
        right = 1000
        bottom = 200

    class HorizontalLayer(ResValues):
        min_height = 5
        max_height = 100
        min_depth = 5
        max_depth = 50

    class InclinedLayer(HorizontalLayer):
        min_angle = 0
        max_angle = 30

    class CurvedLayer(InclinedLayer):
        min_height_curve = 5
        max_height_curve = 15
        min_width_curve = 10
        max_width_curve = 100

    class Wedge(ResValues):
        min_depth = 1
        max_depth = 15

    class InnerWedge(Wedge, ResValues):
        min_height = 10
        max_height = 40
        min_width = 40
        max_width = 300

    class OuterWedge(Wedge, ResValues):
        min_height = 5
        max_height = 100

    class Lens(ResValues):
        min_height = 5
        max_height = 20
        min_width = 10
        max_width = 100
        min_depth = 20
        max_depth = 40

    class LayersContact(CurvedLayer):
        min_height = 10
        max_height = 40
        min_contact_angle = 20
        max_contact_angle = 90


# world = mt.createWorld(start=[ModelConfig.World.x_lt, ModelConfig.World.z_lt],
#                        end=[ModelConfig.World.x_rb, -30],
#                        marker=0
#                        )

#
# height = -rand.randrange(ModelConfig.HorizontalLayer.min_height, ModelConfig.HorizontalLayer.max_height)
# start = -20
#
#
# print(start + height)
# world = world + mt.createPolygon([[ModelConfig.World.x_lt, start],
#                                   [ModelConfig.World.x_rb, start],
#                                   [ModelConfig.World.x_rb, start + height],
#                                   [ModelConfig.World.x_lt, start + height],
#                                   # [ERTModelConfig.World.x_lt, start]
#                                   ], isClosed=True,
#                                  interpolate='spline', marker=1)
#
# rhomap = [[0, 1000],
#           [1, 2000]]
#
# print(world)
#
#
# mesh = mt.createMesh(world, quality=34)
#
# pg.show(mesh, data=rhomap, label=pg.unit('res'), showMesh=True)
# plt.show()


class ResObject:
    left_border_pts = ((ModelConfig.World.left, ModelConfig.World.top),
                       (ModelConfig.World.left, ModelConfig.World.bottom))

    bottom_border_pts = ((ModelConfig.World.left, ModelConfig.World.bottom),
                         (ModelConfig.World.right, ModelConfig.World.bottom))

    right_border_pts = ((ModelConfig.World.right, ModelConfig.World.bottom),
                        (ModelConfig.World.right, ModelConfig.World.top))

    top_border_pts = ((ModelConfig.World.right, ModelConfig.World.top),
                      (ModelConfig.World.left, ModelConfig.World.top))

    def __init__(self,
                 resistivity: Optional[ResType] = None,
                 marker: Optional[int] = None,
                 *args, **kwargs):
        if resistivity is None:
            self.resistivity = rand.randrange(ModelConfig.ResValues.min_res, ModelConfig.ResValues.max_res)
        else:
            self.resistivity = resistivity

        self.marker = marker
        self.mesh = None
        self.random = np.random.uniform

    def _gen_value_if_need(self, min_value: float, max_value: float, curr_value: float, include_negative: bool = False):
        if curr_value is None:
            if include_negative:
                values = [self.random(min_value, max_value), self.random(-max_value, -min_value)]
                return np.random.choice(values)
            else:
                return self.random(min_value, max_value)
        else:
            return curr_value

    def _construct_mesh(self):
        pass

    def get_mesh(self):
        return self.mesh

    def __call__(self):
        return self.get_mesh()

    def set_marker(self, marker: int):
        mt.polytools.setPolyRegionMarker(self.mesh, marker=marker)

    @staticmethod
    def _get_intersection_point(
            line_1_pt_1: PointType,
            line_1_pt_2: PointType,
            line_2_pt_1: PointType,
            line_2_pt_2: PointType,
    ) -> Optional[PointType]:
        """
        Find the intersection point of two lines defined by two points.
        The intersection is also calculated on the continuation of the segments specified by these points.
        """
        s = np.vstack([line_1_pt_1, line_1_pt_2, line_2_pt_1, line_2_pt_2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return None
        return int(x / z), int(y / z)

    @staticmethod
    def _check_point_inside_model(pt: PointType):
        # print(pt)
        # print(ModelConfig.World.left <= pt[0] <= ModelConfig.World.right)
        # print(ModelConfig.World.top <= pt[1] <= ModelConfig.World.bottom)
        if ModelConfig.World.left <= pt[0] <= ModelConfig.World.right and \
                ModelConfig.World.top <= pt[1] <= ModelConfig.World.bottom:
            return True
        else:
            return False

    @classmethod
    def _calc_opposite_point_with_angle(cls, depth: float, pt: PointType, angle: float):
        assert ModelConfig.World.top <= depth <= ModelConfig.World.bottom
        assert cls._check_point_inside_model(pt)
        assert -90 <= angle <= 90

        dx = 1
        dz = - np.tan(np.deg2rad(angle)) * dx

        delta_pt = (pt[0] + dx, pt[1] + dz)

        bottom_intersection = cls._get_intersection_point(pt, delta_pt, *cls.bottom_border_pts)
        print(bottom_intersection)

        if bottom_intersection is None:  # lines are parallel
            pass
        elif cls._check_point_inside_model(bottom_intersection):
            return bottom_intersection
        else:
            pass

        if angle >= 0:
            left_intersection = cls._get_intersection_point(pt, delta_pt, *cls.left_border_pts)

            if cls._check_point_inside_model(left_intersection):
                return left_intersection
            else:
                raise ValueError('Unrecognized point')

        elif -90 <= angle < 0:
            right_intersection = cls._get_intersection_point(pt, delta_pt, *cls.right_border_pts)

            if cls._check_point_inside_model(right_intersection):
                return right_intersection
            else:
                raise ValueError('Unrecognized point')



class World(ResObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._construct_mesh()

    def _construct_mesh(self):
        self.mesh = mt.createWorld(start=[ModelConfig.World.left, ModelConfig.World.top],
                                   end=[ModelConfig.World.right, -ModelConfig.World.bottom],
                                   marker=self.marker)


class HorizontalLayer(ResObject):
    def __init__(self,
                 depth: Optional[float] = None,
                 height: Optional[float] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.height = height

        self._construct_mesh()

    def _construct_mesh(self):
        depth = self._gen_value_if_need(ModelConfig.HorizontalLayer.min_depth,
                                        ModelConfig.HorizontalLayer.max_depth,
                                        self.depth)
        height = self._gen_value_if_need(ModelConfig.HorizontalLayer.min_height,
                                         ModelConfig.HorizontalLayer.max_height,
                                         self.height)

        neg_depth = -depth
        neg_height = -height

        self.mesh = mt.createPolygon([[ModelConfig.World.left, neg_depth],
                                      [ModelConfig.World.right, neg_depth],
                                      [ModelConfig.World.right, neg_depth + neg_height],
                                      [ModelConfig.World.left, neg_depth + neg_height]],
                                     isClosed=True,
                                     interpolate='spline')


class InclinedLayer(ResObject):
    def __init__(self,
                 depth: Optional[float] = None,
                 height: Optional[float] = None,
                 angle: Optional[float] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.height = height
        self.angle = angle

        self._construct_mesh()

    def _construct_mesh(self):
        depth = self._gen_value_if_need(ModelConfig.InclinedLayer.min_depth,
                                        ModelConfig.InclinedLayer.max_depth,
                                        self.depth)
        height = self._gen_value_if_need(ModelConfig.InclinedLayer.min_height,
                                         ModelConfig.InclinedLayer.max_height,
                                         self.height)
        angle = self._gen_value_if_need(ModelConfig.InclinedLayer.min_angle,
                                        ModelConfig.InclinedLayer.max_angle,
                                        self.angle,
                                        True)

        if 0 <= angle <= 90:
            upper_line_pt1 = (ModelConfig.World.right, depth)
            upper_line_pt2 = self._calc_opposite_point_with_angle(depth, upper_line_pt1, angle)
            bottom_value = depth + height / np.cos(np.deg2rad(angle))

            if bottom_value > ModelConfig.World.bottom:
                rb_pt = (ModelConfig.World.right, ModelConfig.World.bottom)
                verts = [upper_line_pt1, rb_pt, upper_line_pt2]
            else:
                lower_line_pt1 = (ModelConfig.World.right, bottom_value)
                lower_line_pt2 = self._calc_opposite_point_with_angle(depth, lower_line_pt1, angle)
                verts = [upper_line_pt1, upper_line_pt2, lower_line_pt2, lower_line_pt1]

        elif -90 <= angle < 0:
            upper_line_pt1 = (ModelConfig.World.left, depth)
            upper_line_pt2 = self._calc_opposite_point_with_angle(depth, upper_line_pt1, angle)
            bottom_value = depth + height / np.cos(np.deg2rad(angle))

            if bottom_value > ModelConfig.World.bottom:
                lb_pt = (ModelConfig.World.left, ModelConfig.World.bottom)
                verts = [upper_line_pt1, lb_pt, upper_line_pt2]
            else:
                lower_line_pt1 = (ModelConfig.World.left, bottom_value)
                lower_line_pt2 = self._calc_opposite_point_with_angle(depth, lower_line_pt1, angle)
                verts = [upper_line_pt1, upper_line_pt2, lower_line_pt2, lower_line_pt1]
        else:
            raise ValueError("Angle must be between -90 and 90")

        verts = [(vert[0], -vert[1]) for vert in verts]

        # if self.marker:
        self.mesh = mt.createPolygon(verts,
                                     isClosed=True,
                                     interpolate='spline', marker=self.marker)




        # print(self._calc_opposite_point_with_angle(depth, (500, depth), 45))
        # print(self._calc_opposite_point_with_angle(depth, (500, depth), -45))



# l = InclinedLayer(angle=89)


world = World(marker=0)

layer2 = InclinedLayer(angle=-3, height=50, marker=1, depth=10)
layer1 = InclinedLayer(angle=-3, height=100, marker=1, depth=20)
# layer1.set_marker(2)


# layer2.set_marker(3)

# layer1 = mt.mergeMeshes([layer1.get_mesh(), layer2.get_mesh()])

# rhomap = [[1, 2000],
#           [2, 3000],
#           [3, 4000]]

def merge(plcs, tol=1e-3):
    plc = pg.Mesh(dim=2, isGeometry=True)

    for p in plcs:
        nodes = []
        for n in p.nodes():
            nn = plc.createNodeWithCheck(n.pos(), tol,
                                         warn=False, edgeCheck=True)
            if n.marker() != 0:
                nn.setMarker(n.marker())
            nodes.append(nn)

        for e in p.boundaries()[::3]:
            plc.createEdge(nodes[e.node(0).id()], nodes[e.node(1).id()],
                           e.marker())

        if len(p.regionMarkers()) > 0:
            for rm in p.regionMarkers():
                plc.addRegionMarker(rm)

        if len(p.holeMarker()) > 0:
            for hm in p.holeMarker():
                plc.addHoleMarker(hm)

    return plc

# plc = world()
# plc = mt.polytools.mergePLC([plc, layer2()])
# plc = mt.polytools.mergePLC([plc, layer1()])


plc = merge([world(), layer1(), layer2()])

fig, ax = pg.plt.subplots()
drawMesh(ax, plc)
drawMesh(ax, mt.createMesh(plc))
pg.wait()







# world = mt.mergePLC([world.get_mesh() + layer1.get_mesh() + layer2.get_mesh()])
#
#
# mesh = mt.createMesh(world, quality=34)
#
# pg.show(mesh, data=rhomap, label=pg.unit('res'), showMesh=True)
# plt.show()


# import pygimli as pg
# import pygimli.meshtools as mt
# from pygimli.viewer.mpl import drawMesh
# world = mt.createWorld(start=[-10, 0], end=[10, -10], marker=0, worldMarker=True)
#
# c1 = mt.createCircle([-1, -4], radius=1.5, area=0.1, segments=5, marker=2)
# # mt.polytools.setPolyRegionMarker(c1, 2)
# c2 = mt.createCircle([-6, -5], radius=[1.5, 3.5], isHole=1, marker=0)
# # mt.polytools.setPolyRegionMarker(c2, 0)
# # r1 = mt.createRectangle(pos=[3, -5], size=[2, 2], marker=3)
# r1 = mt.createRectangle(start=[6, -6], end=[10, -10], marker=10)
# # mt.polytools.setPolyRegionMarker(r1, 10)
#
# r2 = mt.createRectangle(start=[5, -5], end=[9, -9], marker=3)
# # mt.polytools.setPolyRegionMarker(r2, 3)
#
#
# plc = mt.mergePLC([world, c1, c2, r1, r2])
# fig, ax = pg.plt.subplots()
# drawMesh(ax, plc)
# drawMesh(ax, mt.createMesh(plc))
# pg.wait()
