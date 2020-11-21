from typing import Union, Optional, List, Tuple, Iterable
import random
from pathlib import Path

import pygimli as pg
import numpy as np
from pygimli import meshtools as mt
from pygimli.core._pygimli_ import Mesh
import random as rand
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, MultiPolygon, LineString
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
        max_height = 30
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
        min_norm_pos = 0
        max_norm_pos = 1
        min_segments = 3
        max_segments = 15

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
        min_height = 10
        max_height = 30
        min_width = 30
        max_width = 100
        min_depth = 0
        max_depth = 50
        min_x_norm = 0
        max_x_norm = 1
        min_height_truncated_norm = 0.1
        max_height_truncated_norm = 0.9
        min_angle = -30
        max_angle = 30
        allow_symmetric = True

    class BigLens(ResValues):
        """
        The lens represents an object that looks like a truncated ellipse.
        """
        min_height = 15
        max_height = 35
        min_width = 100
        max_width = 1000
        min_depth = 0
        max_depth = 50
        min_x_norm = 0
        max_x_norm = 1
        min_height_truncated_norm = 0.1
        max_height_truncated_norm = 0.9
        min_angle = -5
        max_angle = 5
        allow_symmetric = True

    class LayersContact(CurvedLayer):
        """
        Composite transformation of multiple layers, connecting two layers of different resistance.
        contact_angle: angle of inclination of the contact surface
        """
        min_contact_angle = 20
        max_contact_angle = 90
        min_contact_position_norm = -1
        max_contact_position_norm = 1


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

    def _gen_value_if_need(self,
                           min_value: float,
                           max_value: float,
                           curr_value: float,
                           include_negative: bool = False) -> float:
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

    def is_valid(self) -> bool:
        return not isinstance(self.get_polygon(), OutOfModelObject)

    def _construct_polygon(self):
        """
        An abstract method that must be called at the end of the "__init__" to build the object's polygon.
        """
        pass

    def get_polygon(self) -> Polygon:
        return self.polygon

    def set_marker(self, marker: int) -> None:
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
    def create_pg_world(cls) -> Mesh:
        return mt.createWorld(start=[ModelConfig.World.left, ModelConfig.World.top],
                              end=[ModelConfig.World.right, -ModelConfig.World.bottom],
                              marker=0)

    @classmethod
    def get_world_pts(cls, negative_bottom: bool = True) -> Tuple[Tuple[float, float], ...]:
        """
        The method returns a set of corner points that constrain the dimensions of the model.
        """
        sign = -1 if negative_bottom else 1

        return ((ModelConfig.World.left, ModelConfig.World.top),
                (ModelConfig.World.right, ModelConfig.World.top),
                (ModelConfig.World.right, sign * ModelConfig.World.bottom),
                (ModelConfig.World.left, sign * ModelConfig.World.bottom))

    @staticmethod
    def _check_point_inside_model(pt: PointType) -> bool:
        if ModelConfig.World.left <= pt[0] <= ModelConfig.World.right and \
                ModelConfig.World.top <= pt[1] <= ModelConfig.World.bottom:
            return True
        else:
            return False

    @classmethod
    def _calc_opposite_point_with_angle(cls, pt: PointType, angle: float) -> PointType:
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
    def _finalize_polygon(cls, polygon: Polygon) -> Union[Polygon, OutOfModelObject]:
        world_polygon = Polygon(cls.get_world_pts(negative_bottom=False))
        polygon = world_polygon.intersection(polygon)
        verts = cls.get_values_from_polygon(polygon)

        if len(verts) > 0:
            verts = [(vert[0], -vert[1]) for vert in verts]
            return Polygon(verts)
        else:
            return OutOfModelObject()

    @classmethod
    def show_poly(cls, polygons: Union[Polygon, Iterable[Polygon]]) -> None:
        if not isinstance(polygons, Iterable) and isinstance(polygons, Polygon):
            polygons = [polygons]

        fig, ax = plt.subplots()

        for poly in polygons:
            xy = cls.get_values_from_polygon(poly, as_vectors=True)
            ax.plot(*xy)
        fig.show()

    @classmethod
    def get_values_from_polygon(cls,
                                polygon: Polygon,
                                as_vectors: bool = False) -> \
            Union[Tuple[PointType, ...], Tuple[Tuple[float], Tuple[float]]]:
        coords = polygon.exterior.coords
        if as_vectors:
            return tuple(zip(*coords))
        else:
            return tuple(coords)

    def get_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({'geometry': self.get_polygon(),
                                 'marker': [self.marker]})

    @classmethod
    def calc_distance(cls, pt_1: PointType, pt_2: PointType):
        return np.sqrt((pt_1[0] - pt_2[0]) ** 2 + (pt_1[1] - pt_2[1]) ** 2)

    @classmethod
    def calc_angle(cls, pt_1: PointType, pt_2: PointType) -> float:
        if pt_1[0] == pt_2[0]:
            angle = 90.0
        else:
            tg = (pt_2[1] - pt_1[1]) / (pt_2[0] - pt_1[0])
            angle = np.arccos(np.sqrt(1 / (1 + tg ** 2)))
            angle = np.rad2deg(angle)
        return angle


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
        self.height = self._gen_value_if_need(ModelConfig.HorizontalLayer.min_height,
                                         ModelConfig.HorizontalLayer.max_height, height)
        self._construct_polygon()

    def _construct_polygon(self):
        depth = self._gen_value_if_need(ModelConfig.HorizontalLayer.min_depth,
                                        ModelConfig.HorizontalLayer.max_depth,
                                        self.depth)
        height = self.height

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
        self.height = self._gen_value_if_need(ModelConfig.InclinedLayer.min_height,
                                         ModelConfig.InclinedLayer.max_height, height)
        self._construct_polygon()

    def _construct_polygon(self):
        """
        The method builds a polygon by calculating the coordinates of the top and bottom lines of the layer.
        Since the layer is infinite, it touches one of the boundaries higher than the other.
        """
        depth = self._gen_value_if_need(ModelConfig.InclinedLayer.min_depth,
                                        ModelConfig.InclinedLayer.max_depth,
                                        self.depth)
        height = self.height
        angle = self._gen_value_if_need(ModelConfig.InclinedLayer.min_angle,
                                        ModelConfig.InclinedLayer.max_angle,
                                        self.angle,
                                        True)

        extra_x = max(ModelConfig.World.right, ModelConfig.World.bottom) ** 3

        horizontal_layer = Polygon([[ModelConfig.World.left - extra_x, depth],
                                    [ModelConfig.World.right + extra_x, depth],
                                    [ModelConfig.World.right + extra_x, depth + height],
                                    [ModelConfig.World.left - extra_x, depth + height]])

        inclined_layer = rotate(horizontal_layer, angle)

        if 0 <= angle < 90:
            x_border = ModelConfig.World.left
        elif -90 < angle < 0:
            x_border = ModelConfig.World.right
        else:
            raise ValueError('Angle must be between -90 and 90 except one')

        pt1_border = (x_border, ModelConfig.World.bottom)
        pt2_border = (x_border, ModelConfig.World.top)

        pts_inclined = self.get_values_from_polygon(inclined_layer)
        pt1_upper, pt2_upper = pts_inclined[:2]

        _, z_min = self._get_intersection_point(pt1_border, pt2_border, pt1_upper, pt2_upper)
        delta_z = depth - z_min
        pts_inclined_shifted = [(val[0], val[1] + delta_z) for val in pts_inclined]

        poly_inclined_shifted = Polygon(pts_inclined_shifted)

        self.polygon = self._finalize_polygon(poly_inclined_shifted)


class Buldge(ResObject):
    def __init__(self,
                 depth: Optional[float] = None,
                 height: Optional[float] = None,
                 height_curve: Optional[float] = None,
                 width_curve: Optional[float] = None,
                 angle: Optional[float] = None,
                 seed_buldge: Optional[int] = None,
                 norm_pos: Optional[float] = None,
                 num_segments: Optional[int] = None,
                 *args, **kwargs):
        """
        Class allow you create inclined layer with specified parameters.
        See "args" and "kwargs" description in parent class.
        """
        super().__init__(*args, **kwargs)
        self.angle = angle
        self.depth = depth
        self.height = height
        self.height_curve = height_curve
        self.width_curve = width_curve
        self.seed_buldge = seed_buldge
        self.norm_pos = norm_pos
        self.num_segments = num_segments
        self._max_seed = 10000

        self._construct_polygon()

    def _construct_polygon(self):
        """
        The method builds a polygon by calculating the coordinates of the top and bottom lines of the layer.
        Since the layer is infinite, it touches one of the boundaries higher than the other.
        """
        depth = self._gen_value_if_need(ModelConfig.CurvedLayer.min_depth,
                                        ModelConfig.CurvedLayer.max_depth,
                                        self.depth)
        height = self._gen_value_if_need(ModelConfig.CurvedLayer.min_height,
                                         ModelConfig.CurvedLayer.max_height,
                                         self.height)
        height_curve = self._gen_value_if_need(ModelConfig.CurvedLayer.min_height_curve,
                                               ModelConfig.CurvedLayer.max_height_curve,
                                               self.height_curve)
        width_curve = self._gen_value_if_need(ModelConfig.CurvedLayer.min_width_curve,
                                              ModelConfig.CurvedLayer.max_width_curve,
                                              self.width_curve)
        angle = self._gen_value_if_need(ModelConfig.CurvedLayer.min_angle,
                                        ModelConfig.CurvedLayer.max_angle,
                                        self.angle,
                                        True)
        norm_pos = self._gen_value_if_need(ModelConfig.CurvedLayer.min_norm_pos,
                                           ModelConfig.CurvedLayer.max_norm_pos,
                                           self.norm_pos)
        num_segments = self._gen_value_if_need(ModelConfig.CurvedLayer.min_segments,
                                               ModelConfig.CurvedLayer.max_segments,
                                               self.num_segments)

        if self.seed_buldge is not None:
            seed_buldge = self.seed_buldge
        else:
            seed_buldge = int(self.random(0, self._max_seed))

        extra_x = max(ModelConfig.World.right, ModelConfig.World.bottom) ** 3

        horizontal_layer = Polygon([[ModelConfig.World.left - extra_x, depth],
                                    [ModelConfig.World.right + extra_x, depth],
                                    [ModelConfig.World.right + extra_x, depth + height],
                                    [ModelConfig.World.left - extra_x, depth + height]])

        inclined_layer = rotate(horizontal_layer, angle)

        if 0 <= angle < 90:
            x_border = ModelConfig.World.left
        elif -90 < angle < 0:
            x_border = ModelConfig.World.right
        else:
            raise ValueError('Angle must be between -90 and 90 except one')

        pt1_border = (x_border, ModelConfig.World.bottom)
        pt2_border = (x_border, ModelConfig.World.top)

        pts_inclined = self.get_values_from_polygon(inclined_layer)
        pt1_upper, pt2_upper = pts_inclined[:2]

        _, z_min = self._get_intersection_point(pt1_border, pt2_border, pt1_upper, pt2_upper)
        delta_z = depth - z_min
        pts_inclined_shifted = [(val[0], val[1] + delta_z) for val in pts_inclined]

        poly_inclined_shifted = Polygon(pts_inclined_shifted)

        poly_inclined_shifted = self._finalize_polygon(poly_inclined_shifted)
        self.show_poly(poly_inclined_shifted)

        if isinstance(poly_inclined_shifted, OutOfModelObject):
            self.polygon = OutOfModelObject()
        else:
            pts_inclined_shifted = [(val[0], -val[1]) for val in self.get_values_from_polygon(poly_inclined_shifted)]
            pts_inclined_shifted = np.array(pts_inclined_shifted)

            if angle >= 0:
                z_values = pts_inclined_shifted[:, 1]
                idx_min = z_values == min(z_values)
                top_points = pts_inclined_shifted[idx_min, :]

                x_values = top_points[:, 0]
                idx_max = np.argmax(x_values)
                top_point = top_points[idx_max]

                x_values = pts_inclined_shifted[:, 0]
                idx_max = x_values == max(x_values)
                bottom_points = pts_inclined_shifted[idx_max, :]

                z_values = bottom_points[:, 1]
                idx_min = np.argmin(z_values)
                bottom_point = bottom_points[idx_min]
            else:
                raise NotImplemented()

            prev_seed = np.random.get_state()

            np.random.seed(seed_buldge)
            buldge_pts_x = np.sort(self.random(0, width_curve, int(num_segments)))

            sigma = num_segments / 4
            x_gaus = np.linspace(0, num_segments, num_segments)
            buldge_pts_z = height_curve * np.exp(-((x_gaus - num_segments // 2) / sigma) ** 2)

            delta_random_z = self.random(-height_curve * 0.1, height_curve * 0.1, int(num_segments))

            np.random.set_state(prev_seed)

            buldge_pts_z += delta_random_z

            buldge_pts_x[0] = 0
            buldge_pts_z[0] = 0
            buldge_pts_z[-1] = 0

            buldge_pts_z = np.clip(buldge_pts_z, 0, height_curve)
            self_intersection = buldge_pts_z != 0
            self_intersection[0] = True
            self_intersection[-1] = True

            buldge_pts_x = buldge_pts_x[self_intersection]
            buldge_pts_z = buldge_pts_z[self_intersection]

            pts_buldge = tuple(zip(buldge_pts_x, buldge_pts_z))

            poly_buldge = Polygon(pts_buldge)
            poly_buldge = rotate(poly_buldge, angle, (0, 0))

            x_buldge, z_buldge = np.sum(np.vstack([top_point, bottom_point]), axis=0) * norm_pos

            poly_buldge = translate(poly_buldge, x_buldge, z_buldge + 0.1)

            self.show_poly(poly_buldge)
            pts_inclined_shifted = Polygon(pts_inclined_shifted).union(poly_buldge)

            self.show_poly(self._finalize_polygon(pts_inclined_shifted))


class Lens(ResObject):
    def __init__(self,
                 depth: Optional[float] = None,
                 height: Optional[float] = None,
                 width: Optional[float] = None,
                 x0: Optional[float] = None,
                 angle: Optional[float] = None,
                 trunc: Optional[float] = None,
                 symmetric: Optional[bool] = None,
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
        self.angle = angle
        self.trunc = trunc
        self.symmetric = symmetric

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
        angle = self._gen_value_if_need(ModelConfig.Lens.min_angle,
                                        ModelConfig.Lens.max_angle,
                                        self.angle)
        if self.symmetric is not None:
            symmetric = bool(self.symmetric)
        elif ModelConfig.Lens.allow_symmetric:
            symmetric = True if self.random(0, 1) > 0.5 else False
        else:
            symmetric = False

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

        # TODO: replace h to 2 * h for symmetric
        if symmetric:
            extra_x = x[::-1]
            extra_z = 2 * min(z) - z
            x = np.hstack([x, extra_x])
            z = np.hstack([z, extra_z])

        lens_polygon = Polygon(tuple(zip(x, z)))
        lens_polygon = rotate(lens_polygon, angle, (x_0, depth))

        self.polygon = self._finalize_polygon(lens_polygon)


class BigLens(ResObject):
    def __init__(self,
                 depth: Optional[float] = None,
                 height: Optional[float] = None,
                 width: Optional[float] = None,
                 x0: Optional[float] = None,
                 angle: Optional[float] = None,
                 trunc: Optional[float] = None,
                 symmetric: Optional[bool] = None,
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
        self.angle = angle
        self.trunc = trunc
        self.symmetric = symmetric

        self._construct_polygon()

    def _construct_polygon(self):
        """
        The method builds a polygon by calculating the coordinates of the top and bottom lines of the layer.
        Since the layer is infinite, it touches one of the boundaries higher than the other.
        """
        depth = self._gen_value_if_need(ModelConfig.BigLens.min_depth,
                                        ModelConfig.BigLens.max_depth,
                                        self.depth)
        height = self._gen_value_if_need(ModelConfig.BigLens.min_height,
                                         ModelConfig.BigLens.max_height,
                                         self.height)
        width = self._gen_value_if_need(ModelConfig.BigLens.min_width,
                                        ModelConfig.BigLens.max_width,
                                        self.width)
        x0 = self._gen_value_if_need(ModelConfig.World.right * ModelConfig.BigLens.min_x_norm,
                                     ModelConfig.World.right * ModelConfig.BigLens.max_x_norm,
                                     self.x0)
        trunc = self._gen_value_if_need(ModelConfig.BigLens.min_height_truncated_norm,
                                        ModelConfig.BigLens.max_height_truncated_norm,
                                        self.trunc)
        angle = self._gen_value_if_need(ModelConfig.BigLens.min_angle,
                                        ModelConfig.BigLens.max_angle,
                                        self.angle)
        if self.symmetric is not None:
            symmetric = bool(self.symmetric)
        elif ModelConfig.BigLens.allow_symmetric:
            symmetric = True if self.random(0, 1) > 0.5 else False
        else:
            symmetric = False

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

        # TODO: replace h to 2 * h for symmetric
        if symmetric:
            extra_x = x[::-1]
            extra_z = 2 * min(z) - z
            x = np.hstack([x, extra_x])
            z = np.hstack([z, extra_z])

        lens_polygon = Polygon(tuple(zip(x, z)))
        lens_polygon = rotate(lens_polygon, angle, (x_0, depth))

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
                extra_poly = extra_poly.get_geodataframe()
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


class Compose:
    def __init__(self):
        """In the class constructor, create the depth attribute and the
        compose list that will contain the composition polygons."""
        self.depth = 0
        self.compos = []
        "Creating the possibility of a fault, little lenses, or an angle."
        self.have_angle = np.random.choice([1, 0])
        self.have_lens = np.random.choice([1, 0])
        self.have_fault = np.random.choice([1, 0])
        "Next, we create attributes for the number of layers, big lenses, and little lenses."
        self.num_layers = np.random.randint(1, 5)
        self.num_big_lens = np.random.randint(1, 3)
        self.num_lens = self.have_lens * np.random.randint(1, 5)
        """ Creating attributes for small and large angle.
        A large angle is used in creating a composition with large lenses."""
        self.angle_big = self.have_angle * np.random.uniform(-30, 30)
        self.angle_mini = self.have_angle * np.random.uniform(-5, 5)
        "Choosing a composition."
        self.select = np.random.choice([1, np.random.choice([2, 3, 4])])
        if self.select == 1:
            self.get_layers()
        elif self.select == 2:
            self.get_big_lens()
        elif self.select == 3:
            self.get_lens()
        else:
            self.get_fault()
        "Assign markers to the polygon of the compos list."
        self.get_markers()
        "Take the compass list and create a plc."
        self.final = None
        self.get_plc()
        self.rhomap = None
        self.fill_data()

    @staticmethod
    def create_fault():
        plus = np.random.uniform(60, 80)
        minus = np.random.uniform(-80, -60)
        angle = np.random.choice([plus, minus])
        height = np.random.uniform(10, 500)
        depth = -1000
        fault = InclinedLayer(depth=depth, angle=angle, height=height)
        return fault

    """Objects are created by calling functions that add new polygons to the compos list. 
    Enter in the order in which the polygons should be layered on top of each other."""
    def get_big_lens(self):
        num_big_lens = self.num_big_lens
        for num_big_len in range(num_big_lens):
            big_len = BigLens(angle=self.angle_mini)
            self.compos.append(big_len)

    def get_fault(self):
        self.compos.append(Compose.create_fault())

    def get_lens(self):
        num_lens = self.num_lens
        for num_len in range(num_lens):
            lens = Lens(angle=self.angle_big)
            self.compos.append(lens)

    def get_layers(self):
        num_layers = self.num_layers
        depth = self.depth
        if self.have_fault == 1 and self.have_angle == 0:
            Compose.get_fault(self)
        else:
            pass
        for num_layer in range(num_layers):
            layer = InclinedLayer(depth=depth, angle=self.angle_big, marker=1)
            self.compos.append(layer)
            depth = depth + layer.height
        if self.have_lens == 1:
            self.get_lens()
        else:
            pass

    """The get_markers method measures the length of the compos list 
    and creates a list from 1 to the length of the compos list. 
    Then it randomly sorts the values in the list. At the end, 
    the method assigns a marker from the created list to each polygon in compos."""
    def get_markers(self):
        markers = []
        for i in range(len(self.compos)):
            markers.append(i+1)
        np.random.shuffle(markers)
        for i in range(len(self.compos)):
            self.compos[i].marker = markers[i]

    """The get_markers method measures the length of the campos list and, if it is not empty, 
    creates a shared polygon and retrieves the plc attribute from it. 
    If the list is empty, the class attributes are redefined."""
    def get_plc(self):
        if len(self.compos) > 0:
            self.final = PGMeshCreator(self.compos).plc
        else:
            Compose.__init__(self)

    def fill_data(self):
        rhomap = []
        for i in range(len(self.compos)+1):
            rhomap.append([i, round(np.random.uniform(10, 10000), 2)])
        map = ''
        for i in range(len(rhomap)):
            j = rhomap[i]
            map += str(j[0]) + ' ' + str(j[1]) + ' '
        self.rhomap = map[0: -1]


def create_synthetics(num: Optional[int] = 1):
    for idx in range(num):
        create_sample(idx)


def create_sample(idx: Optional[int] = None):
    project_folder = Path(__file__).resolve().parents[2] / 'project/model_conversion/models'

    np.random.seed(idx)
    random.seed(idx)

    composition = Compose()
    rhomap = composition.rhomap
    plc = composition.final

    plc = mt.createMesh(plc, quality=10)
    plc.save(str(project_folder / ("mesh_" + str(idx + 1))))
    file = open(project_folder / ("map_" + str(idx + 1) + ".txt"), "w")
    file.write(rhomap)
    file.close()

    # os.chdir('..')
    # fig, ax = pg.plt.subplots()
    # drawMesh(ax, plc)
    # drawMesh(ax, mt.createMesh(plc))
    # pg.wait()


if __name__ == "__main__":
    # from pygimli.viewer.mpl import drawMesh
    # from project.config import common_config
    # import os
    # import random
    # from tqdm import tqdm
    # from pygimli import meshtools as msh
    # import pygimli.physics.ert as ert




    # create_synthetics(num=20)
    # create_sample(0)

    ids = list(range(1000))

    from python_utils.runner import Runner

    runner = Runner('process', 16)
    runner(create_sample, ids)


    # file = Path('F:\PycharmProjects\er_dl\project\model_conversion\models\mesh_1.bms')
    # print(file.stat().st_size / 1024)
    # plc = pg.meshtools.readPLC('F:\PycharmProjects\er_dl\project\synthetic\models\mesh_1.bms')
    # print(plc)
    # fig, ax = pg.plt.subplots()
    # drawMesh(ax, plc)
    # # drawMesh(ax, mt.createMesh(plc))
    # pg.wait()
