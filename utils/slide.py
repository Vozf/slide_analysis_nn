from concurrent.futures import ThreadPoolExecutor
from os.path import basename, join

import numpy as np
from matplotlib import pyplot as plt
from openslide import open_slide
from shapely.geometry import Polygon, MultiPolygon, box

from train.datasets_preparation.settings import (
    DEFAULT_CLASS_NAME,
    BACKGROUND_CLASS_NAME,
    UNLABELED_IMAGES_DIR,
    LABELED_IMAGES_DIR
)
from utils.constants import TILE_SIZE, TILE_STEP, AREA_PROCESSING_MULTIPLIER
from utils.functions import dict_assign


class Slide:
    def __init__(self, slide_path):
        self.slide_path = slide_path
        self.slide = open_slide(slide_path)

    def cut_tile(self, x, y, width=TILE_SIZE, height=TILE_SIZE):
        return self.slide.read_region((x, y), 0, (width, height))

    def cut_polygons_data(self, bounding_box_polygons, draw_invalid_polygons=False):
        print('processing slide {}'.format(self.slide_path))

        shapely_poly = [self._create_shapely_polygon(poly, draw_invalid_polygons) for poly in
                        bounding_box_polygons]
        global_multipolygon = MultiPolygon(filter(None, np.hstack(shapely_poly)))

        with ThreadPoolExecutor() as executor:
            dicts = list(executor.map(
                lambda current_polygon: self._process_polygon(current_polygon, global_multipolygon),
                global_multipolygon))

        return dict_assign({}, *dicts)

    def _create_shapely_polygon(self, polygon, draw_invalid_polygons=False):
        shapely_polygon = Polygon(polygon)
        if shapely_polygon.is_valid:
            return shapely_polygon

        fixed_shapely_polygon = shapely_polygon.buffer(0)

        if draw_invalid_polygons:
            plt.scatter(*np.asarray(polygon).T, c='r', s=200)

            if hasattr(fixed_shapely_polygon, 'exterior'):
                plt.scatter(*np.array(fixed_shapely_polygon.exterior.coords.xy), c='g')
            else:
                [plt.scatter(*np.array(poly.exterior.coords.xy), c='g') for poly in
                 fixed_shapely_polygon]

            plt.show()

        if fixed_shapely_polygon.is_valid:
            return fixed_shapely_polygon

        print('invalid polygon at path: {}'.format(self.slide_path))

        return None

    def _process_polygon(self, current_polygon, global_multipolygon):
        print(self._get_bounding_box_for_polygon(current_polygon), self._get_processing_area_for_polygon(current_polygon))
        dictionary = {}

        x1pa, y1pa, x2pa, y2pa = self._get_processing_area_for_polygon(current_polygon)

        # if x2pa - x1pa > 10000:
        #     return {}

        # self._save_tile((x1pa, y1pa, x2pa, y2pa), dir_path=SMALL_WITH_TUMOR_IMAGES_DIR, ext='tif')
        x = range(x1pa, x2pa, TILE_STEP)
        y = range(y1pa, y2pa, TILE_STEP)
        coords_to_extract = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

        with ThreadPoolExecutor() as executor:
            path_and_classes = executor.map(
                lambda coords: self.save_training_example(*coords, global_multipolygon), coords_to_extract)

        for class_name, image_path in path_and_classes:
            dictionary[image_path] = class_name

        return dictionary

    def save_training_example(self, x_coord, y_coord, global_multipolygon):
        tile_box = (x_coord, y_coord, x_coord + TILE_SIZE, y_coord + TILE_SIZE)
        if self._is_intersected(global_multipolygon, tile_box=tile_box):
            dir = LABELED_IMAGES_DIR
            class_name = DEFAULT_CLASS_NAME

        else:
            dir = UNLABELED_IMAGES_DIR
            class_name = BACKGROUND_CLASS_NAME

        image_path = self._save_tile(tile_box, dir_path=dir)
        return class_name, image_path

    def _get_processing_area_for_polygon(self, polygon):
        x1, y1, x2, y2 = self._get_bounding_box_for_polygon(polygon)

        # enlarge_area_x = int(AREA_PROCESSING_MULTIPLIER * TILE_SIZE)
        # enlarge_area_y = int(AREA_PROCESSING_MULTIPLIER * TILE_SIZE)
        enlarge_area_x = int((x2 - x1) * (AREA_PROCESSING_MULTIPLIER - 1)/2)
        enlarge_area_y = int((y2 - y1) * (AREA_PROCESSING_MULTIPLIER - 1)/2)

        return max(x1 - enlarge_area_x, 0), \
               max(y1 - enlarge_area_y, 0), \
               min(x2 + enlarge_area_x, self.slide.dimensions[0]), \
               min(y2 + enlarge_area_y, self.slide.dimensions[1])

    def _is_intersected(self, polygon, tile_box):
        rect = box(*tile_box)
        return polygon.intersects(rect)

    def _calculate_local_bounding_boxes(self, bbox, tile_box):
        poly = box(*tile_box).intersection(bbox)
        x1, y1, x2, y2 = tile_box
        global_boxes = self._get_bounding_boxes_for_geometry(poly)
        local_boxes = [(x1g - x1, y1g - y1, x2g - x1, y2g - y1) for x1g, y1g, x2g, y2g in global_boxes]
        return local_boxes

    def _get_bounding_box_for_polygon(self, poly):
        points = np.array(poly.exterior.coords.xy).astype(np.int)
        return points[0].min(), points[1].min(), points[0].max(), points[1].max()

    def _get_bounding_boxes_for_geometry(self, geo):
        try:
            geoms = [geo] if hasattr(geo, 'exterior') else\
                filter(lambda x: hasattr(x, 'exterior'), geo.geoms)

            return list(map(self._get_bounding_box_for_polygon, geoms))
        except:
            print('Unexpected type:',type(geo))
            return []

    def _save_tile(self, tile_box, dir_path, ext='png'):
        tile = self.cut_tile(tile_box[0], tile_box[1], tile_box[2] - tile_box[0],
                             tile_box[3] - tile_box[1])

        image_name = "{0}_({1},{2},{3},{4}).{5}".format(basename(self.slide_path), *tile_box, ext)
        image_path = join(dir_path, image_name)

        tile.save(image_path)
        return image_path
