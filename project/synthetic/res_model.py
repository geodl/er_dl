from typing import Union, Optional, List, Tuple

import pygimli as pg
import numpy as np
from pygimli import meshtools as mt
import random as rand
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd


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

        if marker is not None:
            assert isinstance(marker, int)
            assert marker >= 1

        self.marker = marker
        self.polygon = None
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

    def get_polygon(self):
        return self.polygon

    def set_marker(self, marker: int):
        self.marker = marker

    def __call__(self):
        return self.get_polygon()

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

    @classmethod
    def create_pg_world(cls):
        return mt.createWorld(start=[ModelConfig.World.left, ModelConfig.World.top],
                              end=[ModelConfig.World.right, -ModelConfig.World.bottom],
                              marker=0)

    @classmethod
    def get_world_pts(cls):
        return [[ModelConfig.World.left, ModelConfig.World.top],
                [ModelConfig.World.right, ModelConfig.World.top],
                [ModelConfig.World.right, -ModelConfig.World.bottom],
                [ModelConfig.World.left, -ModelConfig.World.bottom]]

    @staticmethod
    def _check_point_inside_model(pt: PointType):
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
        self.polygon = Polygon(self.get_world_pts())


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

        self.polygon = Polygon([[ModelConfig.World.left, neg_depth],
                                [ModelConfig.World.right, neg_depth],
                                [ModelConfig.World.right, neg_depth + neg_height],
                                [ModelConfig.World.left, neg_depth + neg_height]])


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
        self.polygon = Polygon(verts)


class PGMeshCreator:
    def __init__(self, resobjects_list: Union[ResObject, List[ResObject]]):
        self.objects_list = None
        self.plc = None

        if resobjects_list:
            self.compose_objects(resobjects_list)

    def compose_objects(self, resobjects_list: Union[ResObject, List[ResObject]]):
        if isinstance(resobjects_list, ResObject):
            resobjects_list = [resobjects_list]

        world_polygon = Polygon(ResObject.get_world_pts())

        world = gpd.GeoDataFrame({'geometry': world_polygon, 'marker': [0]})
        first_poly = gpd.GeoDataFrame({'geometry': resobjects_list[0].get_polygon(),
                                      'marker': [resobjects_list[0].marker]})

        res_union = self.merge_two_polygons(world, first_poly)

        if len(resobjects_list) > 1:
            for extra_poly in resobjects_list[1:]:
                extra_poly = gpd.GeoDataFrame({'geometry': extra_poly.get_polygon(),
                                               'marker': [extra_poly.marker]})
                res_union = self.merge_two_polygons(res_union, extra_poly)

        mesh_list = [ResObject.create_pg_world()]

        for _, rows in res_union.iterrows():
            if int(rows['marker']) != 0:
                verts = tuple(zip(*rows['geometry'].exterior.xy))[:-1]
                mesh_list.append(mt.createPolygon(verts, isClosed=True, marker=int(rows['marker'])))

        self.objects_list = resobjects_list
        self.plc = mt.polytools.mergePLC(mesh_list)

        return mt.polytools.mergePLC(mesh_list)

    @staticmethod
    def merge_two_polygons(background_polygons, extra_polygon):
        res_union = gpd.overlay(background_polygons, extra_polygon, how='union')

        res_polygons = []
        res_markers = []

        for _, row in res_union.iterrows():
            if isinstance(row['geometry'], MultiPolygon):
                polygons = list(row['geometry'])
            else:
                polygons = [row['geometry']]

            back_marker = row['marker_1']
            union_marker = row['marker_2']
            extra_marker = extra_polygon['marker'].iloc[0]

            if union_marker == extra_marker:
                res_markers += [extra_marker] * len(polygons)
            else:
                res_markers += [back_marker] * len(polygons)

            res_polygons += polygons

        return gpd.GeoDataFrame({'geometry': res_polygons, 'marker': res_markers})


if __name__ == "__main__":
    from pygimli.viewer.mpl import drawMesh

    layer2 = InclinedLayer(angle=2, height=20, marker=1, depth=20)
    layer1 = InclinedLayer(angle=-12, height=60, marker=2, depth=20)

    mesh = PGMeshCreator([layer1, layer2])
    plc = mesh.plc

    fig, ax = pg.plt.subplots()
    drawMesh(ax, plc)
    drawMesh(ax, mt.createMesh(plc))
    pg.wait()
