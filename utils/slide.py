from collections import defaultdict

from functools import reduce
from shapely.geometry import Polygon, Point, box

from openslide import open_slide

from train.datasets_preparation.settings import (
    DEFAULT_CLASS_NAME,
    UNLABELED_IMAGES_DIR,
    LABELED_IMAGES_DIR)
from utils.constants import TILE_SIZE, TILE_STEP, AREA_PROCESSING_MULTIPLIER
import numpy as np
from os.path import basename, join
from train.datasets_preparation.utils import Label


class Slide:
    def __init__(self, slide_path):
        self.slide_path = slide_path
        self.slide = open_slide(slide_path)

    def cut_tile(self, x, y, width=TILE_SIZE, height=TILE_SIZE):
        return self.slide.read_region((x, y), 0, (width, height))

    def cut_polygons_data(self, bounding_box_polygons):

        shapely_bounding_box_polygons = \
            list(filter(lambda x: x.is_valid, [Polygon(bbox) for bbox in bounding_box_polygons]))

        dicts = list(map(self._process_polygon, shapely_bounding_box_polygons))
        print('dicts')

        labeled_dict, unlabeled_dict = \
            reduce(lambda acc, dic: ({**acc[0], **dic[0]}, {**acc[1], **dic[1]}), dicts,
                   (defaultdict(list), defaultdict(list)))
        print('bue')

        return labeled_dict, unlabeled_dict

    def cut_unlabeled_data(self, number, dir_path):
        pass

    def cut_for_predict(self):
        pass

    def _process_polygon(self, polygon):
        print(self._get_bounding_box_for_polygon(polygon), self._get_processing_area_for_polygon(polygon))
        labeled_dict = defaultdict(list)
        unlabeled_dict = defaultdict(list)

        x1pa, y1pa, x2pa, y2pa = self._get_processing_area_for_polygon(polygon)

        for x_coord in range(x1pa, x2pa, TILE_STEP):
            for y_coord in range(y1pa, y2pa, TILE_STEP):
                tile = self.cut_tile(x_coord, y_coord)
                tile_box = (x_coord, y_coord, x_coord + TILE_SIZE, y_coord + TILE_SIZE)

                if self._is_intersected(polygon, tile_box=tile_box):
                    bounding_boxes = self._calculate_local_bounding_boxes(polygon, tile_box)
                    dir = LABELED_IMAGES_DIR
                    dictionary = labeled_dict
                    class_name = DEFAULT_CLASS_NAME

                else:
                    bounding_boxes = [(None, None, None, None)]
                    dir = UNLABELED_IMAGES_DIR
                    dictionary = unlabeled_dict
                    class_name = None

                image_path = self._save_tile(tile, tile_box=tile_box, dir_path=dir)

                for bbox in bounding_boxes:

                    x1bb, y1bb, x2bb, y2bb = bbox
                    dictionary[image_path].append(
                        Label(
                            path=image_path,
                            x1=x1bb,
                            y1=y1bb,
                            x2=x2bb,
                            y2=y2bb,
                            class_name=class_name))

        return labeled_dict, unlabeled_dict


    def _get_processing_area_for_polygon(self, polygon):
        x1, y1, x2, y2 = self._get_bounding_box_for_polygon(polygon)

        enlarge_area = int(AREA_PROCESSING_MULTIPLIER * TILE_SIZE)

        return max(x1 - enlarge_area, 0), \
               max(y1 - enlarge_area, 0), \
               min(x2 + enlarge_area, self.slide.dimensions[0]), \
               min(y2 + enlarge_area, self.slide.dimensions[1])

    def _is_intersected(self, polygon, tile_box):
        rect = box(*tile_box)
        return polygon.intersects(rect)
        # x1, y1, x2, y2 = tile_box
        # points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        # return any(map(lambda point_coords: polygon.contains(Point(*point_coords)), points))


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





    def _save_tile(self, tile, tile_box, dir_path):
        image_name = "{0}_{1}:{2}:{3}:{4}.png".format(basename(self.slide_path), *tile_box)
        image_path = join(dir_path, image_name)

        tile.save(image_path, 'PNG')
        return image_path


def cut_image_to_tiles(path, bounding_boxes=[], labeled_images_path='', unlabeled_images_path='',
                       tile_size=TILE_SIZE, tile_step=TILE_STEP):
    slide = open_slide(path).read_region((50800, 165600), 0,
                                         (tile_size, tile_size)).save('1.png', 'PNG')

# cut_image_to_tiles('/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/slide_images/Tumor_015.tif')
