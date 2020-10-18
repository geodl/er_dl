from typing import Union, Optional, List, Tuple, Iterable

import pygimli as pg
import numpy as np
from pygimli import meshtools as mt
import random as rand
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import matplotlib.pyplot as plt


ResType = Union[float, np.ndarray]
PointType = Union[Tuple[float, float], np.ndarray]  # point coordinates sorted in (x, y) order


class ModelConfig:
    """
    A class with parameter limits for random generation.
    Also contains the geometrical dimensions of the world.
    """
    class ResValues:
        """
        Resistance limits for all subsequent classes.
        """
        min_res = 10
        max_res = 10000

    class ObjectGrid:
        """
        Discretize smooth objects with a step of "delta" meters.
        """
        delta = 10

    class World(ResValues):
        """
        Model world/background/underlay with dimensions.
        For convenience of calculation, the depth ("bottom") is positive, but at the last step of combining the
        primitives, the depths become negative.
        """
        left = 0
        top = 0
        right = 1000
        bottom = 200

    class HorizontalLayer(ResValues):
        """
        Simple horizontal layer
        height: thickness of layer in orthogonal direction
        depth: depth of top of layer
        """
        min_height = 10
        max_height = 100
        min_depth = 5
        max_depth = 50

    class InclinedLayer(HorizontalLayer):
        """
        Inclined layer of constant thickness.
        depth: the topmost point of the layer, which is on the far left or right of the model
        angle: layer tilt angle. May be negative
        """
        min_angle = 0
        max_angle = 30

    class CurvedLayer(InclinedLayer):
        """
        Inclined layer with a bulge.
        height_curve: the height of the bulge
        width_curve: the width of the bulge
        """
        min_height_curve = 5
        max_height_curve = 15
        min_width_curve = 10
        max_width_curve = 100

    class Wedge(ResValues):
        """
        Basic class for wedges.
        depth: the topmost point of the object, which is on the far left or right of the model
        """
        min_depth = 1
        max_depth = 15

    class InnerWedge(Wedge, ResValues):
        """
        Wedge, the corner of which is inside the model.
        height: layer thickness at the border of the model.
                The thickness gradually decreases from this value at the edge of the model to zero at the corner of
                the wedge.
        width: distance between the border of the model and the corner of the wedge.
        """
        min_height = 10
        max_height = 40
        min_width = 40
        max_width = 300

    class OuterWedge(Wedge, ResValues):
        """
        Wedge, the corner of which is outside the model.
        height: layer thickness at the border of the model.
                The thickness gradually changes from one value at one border to another value at the other border.
        """
        min_height = 5
        max_height = 100

    class Lens(ResValues):
        """
        The lens represents an object that looks like a truncated ellipse.
        """
        min_height = 5
        max_height = 20
        min_width = 10
        max_width = 100
        min_depth = 20
        max_depth = 40
        min_x_norm = 0
        max_x_norm = 1
        min_height_truncated_norm = 0.1
        max_height_truncated_norm = 0.7

    class LayersContact(CurvedLayer):
        """
        Composite transformation of multiple layers, connecting two layers of different resistance.
        height: the thickness of the layers to be joined
        contact_angle: angle of inclination of the contact surface
        """
        min_height = 10
        max_height = 40
        min_contact_angle = 20
        max_contact_angle = 90


class OutOfModelObject:
    pass


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
        """
        Base class for generating objects.
        Contains abstract methods for working with polygons, point validation, and also contains helper methods
        for geometric calculations.

        Class allow you create geometric objects with specified extra parameters.
        If some parameters are not specified, random values will be generated for them,
        taken from the configuration class.

        Arguments:
            resistivity: Resistance values for an object. If not specified, it will be generated randomly.
            marker: A number that will later be assigned the resistance value for all such numbers.

        Attributes:
            resistivity: see Arguments.
            marker: see Arguments.
            polygon: polygon (enclosed area) is a geometric representation of a geoelectric feature.
            It can be represented as a collection of points.
            random: function that generates random values in a given range.
        """
        self.random = np.random.uniform

        if resistivity is None:
            self.resistivity = self.random(ModelConfig.ResValues.min_res, ModelConfig.ResValues.max_res)
        else:
            self.resistivity = resistivity

        if marker is not None:
            assert isinstance(marker, int)
            assert marker >= 1

        self.marker = marker
        self.polygon = None

    def _gen_value_if_need(self, min_value: float, max_value: float, curr_value: float, include_negative: bool = False):
        """
        Method check that "curr_value" is None. If True it generates random value between "min_value" and "max_value",
        including negative ones if "include_negative" is True. Otherwise, it returns the "curr_value".
        """
        if curr_value is None:
            if include_negative:
                values = [self.random(min_value, max_value), self.random(-max_value, -min_value)]
                return np.random.choice(values)
            else:
                return self.random(min_value, max_value)
        else:
            return curr_value

    def is_valid(self):
        return not isinstance(self.get_polygon(), OutOfModelObject)

    def _construct_polygon(self):
        """
        An abstract method that must be called at the end of the "__init__" to build the object's polygon.
        """
        pass

    def get_polygon(self):
        return self.polygon

    def set_marker(self, marker: int):
        """
        The method allows you to change or assign a marker after creating an object.
        """
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
    def get_world_pts(cls, negative_bottom: bool = True):
        """
        The method returns a set of corner points that constrain the dimensions of the model.
        """
        sign = -1 if negative_bottom else 1

        return [[ModelConfig.World.left, ModelConfig.World.top],
                [ModelConfig.World.right, ModelConfig.World.top],
                [ModelConfig.World.right, sign * ModelConfig.World.bottom],
                [ModelConfig.World.left, sign * ModelConfig.World.bottom]]

    @staticmethod
    def _check_point_inside_model(pt: PointType):
        if ModelConfig.World.left <= pt[0] <= ModelConfig.World.right and \
                ModelConfig.World.top <= pt[1] <= ModelConfig.World.bottom:
            return True
        else:
            return False

    @classmethod
    def _calc_opposite_point_with_angle(cls, pt: PointType, angle: float):
        """
        Finds a point on a boundary (left, right or bottom) that lies on a ray passing at a specified "angle" from
        the point "pt".
        """
        assert cls._check_point_inside_model(pt)
        assert -90 <= angle <= 90

        dx = 1
        dz = - np.tan(np.deg2rad(angle)) * dx
        # Extra point on same ray
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

    @classmethod
    def _finalize_polygon(cls, polygon: Polygon):
        world_polygon = Polygon(cls.get_world_pts(negative_bottom=False))
        polygon = world_polygon.intersection(polygon)
        verts = cls.get_values_from_polygon(polygon)

        if len(verts) > 0:
            verts = [(vert[0], -vert[1]) for vert in verts]
            return Polygon(verts)
        else:
            return OutOfModelObject()

    @classmethod
    def show_poly(cls, polygons: Union[Polygon, Iterable[Polygon]]):
        if not isinstance(polygons, Iterable) and isinstance(polygons, Polygon):
            polygons = [polygons]

        fig, ax = plt.subplots()

        for poly in polygons:
            xy = cls.get_values_from_polygon(poly, as_vectors=True)
            ax.plot(*xy)
        fig.show()

    @classmethod
    def get_values_from_polygon(cls, polygon: Polygon, as_vectors: bool = False):
        coords = polygon.exterior.coords
        if as_vectors:
            return tuple(zip(*coords))
        else:
            return tuple(coords)


class World(ResObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._construct_polygon()

    def _construct_polygon(self):
        self.polygon = Polygon(self.get_world_pts())


class HorizontalLayer(ResObject):
    def __init__(self,
                 depth: Optional[float] = None,
                 height: Optional[float] = None,
                 *args, **kwargs):
        """
        Class allow you create horizontal layer with specified parameters.

        See "args" and "kwargs" description in parent class.
        """
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.height = height

        self._construct_polygon()

    def _construct_polygon(self):
        depth = self._gen_value_if_need(ModelConfig.HorizontalLayer.min_depth,
                                        ModelConfig.HorizontalLayer.max_depth,
                                        self.depth)
        height = self._gen_value_if_need(ModelConfig.HorizontalLayer.min_height,
                                         ModelConfig.HorizontalLayer.max_height,
                                         self.height)

        horizontal_layer = Polygon([[ModelConfig.World.left, depth],
                                    [ModelConfig.World.right, depth],
                                    [ModelConfig.World.right, depth + height],
                                    [ModelConfig.World.left, depth + height]])

        self.polygon = self._finalize_polygon(horizontal_layer)


class InclinedLayer(ResObject):
    def __init__(self,
                 depth: Optional[float] = None,
                 height: Optional[float] = None,
                 angle: Optional[float] = None,
                 *args, **kwargs):
        """
        Class allow you create inclined layer with specified parameters.

        See "args" and "kwargs" description in parent class.
        """
        super().__init__(*args, **kwargs)
        self.angle = angle
        self.depth = depth
        self.height = height

        self._construct_polygon()

    def _construct_polygon(self):
        """
        The method builds a polygon by calculating the coordinates of the top and bottom lines of the layer.
        Since the layer is infinite, it touches one of the boundaries higher than the other.
        """
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
            x_pos = ModelConfig.World.right
            scalar_dx = -1
        elif -90 <= angle < 0:
            x_pos = ModelConfig.World.left
            scalar_dx = 1
        else:
            raise ValueError("Angle must be between -90 and 90")

        dx = scalar_dx * max(ModelConfig.World.right, ModelConfig.World.left) ** 2
        dz = - np.tan(np.deg2rad(angle)) * dx

        upper_line_pt1 = (x_pos, depth)
        upper_line_pt2 = (upper_line_pt1[0] + dx, upper_line_pt1[1] + dz)

        lower_line_pt1 = (x_pos, depth + height / np.cos(np.deg2rad(angle)))
        lower_line_pt2 = (lower_line_pt1[0] + dx, lower_line_pt1[1] + dz)

        layer_polygon = Polygon([upper_line_pt1, upper_line_pt2, lower_line_pt2, lower_line_pt1])

        self.polygon = self._finalize_polygon(layer_polygon)


class Lens(ResObject):
    def __init__(self,
                 depth: Optional[float] = None,
                 height: Optional[float] = None,
                 width: Optional[float] = None,
                 x0: Optional[float] = None,
                 trunc: Optional[float] = None,
                 *args, **kwargs):
        """
        Class allow you create inclined layer with specified parameters.

        See "args" and "kwargs" description in parent class.
        """
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.height = height
        self.width = width
        self.x0 = x0
        self.trunc = trunc

        self._construct_polygon()

    def _construct_polygon(self):
        """
        The method builds a polygon by calculating the coordinates of the top and bottom lines of the layer.
        Since the layer is infinite, it touches one of the boundaries higher than the other.
        """
        depth = self._gen_value_if_need(ModelConfig.Lens.min_depth,
                                        ModelConfig.Lens.max_depth,
                                        self.depth)
        height = self._gen_value_if_need(ModelConfig.Lens.min_height,
                                         ModelConfig.Lens.max_height,
                                         self.height)
        width = self._gen_value_if_need(ModelConfig.Lens.min_width,
                                        ModelConfig.Lens.max_width,
                                        self.width)
        x0 = self._gen_value_if_need(ModelConfig.World.right * ModelConfig.Lens.min_x_norm,
                                     ModelConfig.World.right * ModelConfig.Lens.max_x_norm,
                                     self.x0)
        trunc = self._gen_value_if_need(ModelConfig.Lens.min_height_truncated_norm,
                                        ModelConfig.Lens.max_height_truncated_norm,
                                        self.trunc)

        width /= 2

        w_0 = width / np.sqrt(1 - trunc ** 2)
        h_0 = height / (1 - trunc)

        x_0 = x0
        z_0 = depth - h_0 * trunc

        x_l = x_0 - w_0 * np.sqrt(1 - trunc ** 2)
        x_r = x_0 + w_0 * np.sqrt(1 - trunc ** 2)

        # segments each "ModelConfig.ObjectGrid.delta" meters or minimum 5 parts
        num_segments = max(int(x_r - x_l) // ModelConfig.ObjectGrid.delta, 5)
        x = np.linspace(x_l, x_r, num_segments)
        z = z_0 + h_0 * np.sqrt(1 - ((x - x_0) / w_0) ** 2)

        lens_polygon = Polygon(tuple(zip(x, z)))

        self.polygon = self._finalize_polygon(lens_polygon)


class PGMeshCreator:
    def __init__(self, resobjects_list: Union[ResObject, List[ResObject]]):
        """
        The class allows you to combine different primitives on one pygimli Mesh.
        Primitives must inherit from class ResObject, have a polygon and a marker.
        """
        self.objects_list = None
        self.plc = None

        if resobjects_list:
            self.compose_objects(resobjects_list)

    def compose_objects(self, resobjects_list: Union[ResObject, List[ResObject]]):
        """
        The method combines primitives and puts them on one base.

        NOTE:
            The primitives are stacked on top of each other with overlap.
            The number of formed polygons is equal to or greater than the number of the original primitives.
        """

        if isinstance(resobjects_list, (ResObject, OutOfModelObject)):
            resobjects_list = [resobjects_list]

        world_polygon = Polygon(ResObject.get_world_pts())
        res_union = gpd.GeoDataFrame({'geometry': world_polygon, 'marker': [0]})

        for extra_poly in resobjects_list:
            assert extra_poly.marker is not None
            if extra_poly.is_valid():
                extra_poly = gpd.GeoDataFrame({'geometry': extra_poly.get_polygon(),
                                               'marker': [extra_poly.marker]})
                res_union = self.merge_two_polygons(res_union, extra_poly)

        mesh_list = [ResObject.create_pg_world()]

        for _, rows in res_union.iterrows():
            if int(rows['marker']) != 0:
                verts = tuple(rows['geometry'].exterior.coords)[:-1]
                mesh_list.append(mt.createPolygon(verts, isClosed=True, marker=int(rows['marker'])))

        self.objects_list = resobjects_list
        self.plc = mt.polytools.mergePLC(mesh_list)

        return mt.polytools.mergePLC(mesh_list)

    @staticmethod
    def merge_two_polygons(background_polygons, extra_polygon):
        """
        The method looks for the union of polygons of two frames and, based on them, makes a series of new polygons.
        If any final polygon contains the value of the upper polygon, then the marker of this polygon is assigned,
        otherwise the marker of the previous polygon.
        """
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
    from project.config import common_config

    rand.seed(1)
    np.random.seed(1)

    layer1 = InclinedLayer(angle=-12, height=50, marker=1, depth=20)
    layer2 = InclinedLayer(angle=2, height=20, marker=2, depth=20)
    layer3 = InclinedLayer(angle=0, height=70, marker=3, depth=120)
    layer4 = Lens(marker=3, width=150, height=50, x0=300, depth=30)
    layer5 = Lens(marker=1, width=250, height=20, x0=700, depth=150)

    mesh = PGMeshCreator([layer1, layer2, layer3, layer4, layer5])
    plc = mesh.plc

    pg.meshtools.exportPLC(plc, common_config.root_dir / 'exmaple.dat')

    fig, ax = pg.plt.subplots()
    drawMesh(ax, plc)
    drawMesh(ax, mt.createMesh(plc))
    pg.wait()
