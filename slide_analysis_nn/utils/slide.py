import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from matplotlib import pyplot as plt
from openslide import open_slide
from os.path import basename, join
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import scale
import pandas as pd

from slide_analysis_nn.train.datasets_preparation.settings import (
    DEFAULT_CLASS_NAME,
    BACKGROUND_CLASS_NAME,
    SOURCE_PATH,
)
from slide_analysis_nn.train.settings import (
    AREA_PROCESSING_MULTIPLIER,
    MAX_TILES_PER_TUMOR,
    AREA_TO_INTERSECT_MULTIPLIER,
    NETWORK_INPUT_SHAPE,
)
from slide_analysis_nn.utils.ASAP_xml import append_polygons_to_existing_xml
from slide_analysis_nn.tile import TILE_SIZE, TILE_STEP


class Slide:
    def __init__(self, slide_path, draw_invalid_polygons=False, create_xml_with_cut_tiles=False):
        self.slide_path, self.draw_invalid_polygons, self.create_xml_with_cut_tiles = \
            slide_path, draw_invalid_polygons, create_xml_with_cut_tiles

        self.slide = open_slide(slide_path)

    def cut_tile(self, x, y, width=TILE_SIZE, height=TILE_SIZE):
        return self.slide.read_region((x, y), 0, (width, height))

    def cut_polygons_data(self, asap_polygons):
        if not asap_polygons:
            return pd.DataFrame()

        print('processing slide {}'.format(self.slide_path))

        shapely_poly = [self._create_shapely_polygon(poly) for poly in
                        asap_polygons]
        global_multipolygon = MultiPolygon(filter(None, np.hstack(shapely_poly)))

        dfs = [self._process_polygon(current_polygon, global_multipolygon)
               for current_polygon in global_multipolygon]

        slide_df = pd.concat(dfs)
        slide_df['slide_path'] = self.slide_path

        if self.create_xml_with_cut_tiles:
            self._create_xml_annotation_with_marked_tiles(slide_df)

        return slide_df

    def _create_xml_annotation_with_marked_tiles(self, df):
        if df.empty:
            return

        xml_path = '{}_cut.xml'.format(os.path.splitext(self.slide_path)[0])
        truth_xml_path = '{}.xml'.format(os.path.splitext(self.slide_path)[0])

        def get_polygons(class_name):
            tile = df.where(df.class_name == class_name)[['x1', 'y1', 'x2', 'y2']].dropna().values
            polygon = np.concatenate((tile[..., None, :2], tile[..., None, 2:]), axis=1)
            return polygon

        labeled = get_polygons(DEFAULT_CLASS_NAME)
        unlabeled = get_polygons(BACKGROUND_CLASS_NAME)

        append_polygons_to_existing_xml(np.concatenate((labeled, unlabeled)),
                                        predicted_labels=[1] * (len(labeled)) + [0] * len(
                                            unlabeled),
                                        scores=[1] * (len(labeled)) + [0] * len(unlabeled),
                                        source_xml_path=truth_xml_path,
                                        output_xml_path=xml_path)

    def _create_shapely_polygon(self, polygon):
        shapely_polygon = Polygon(polygon)
        if shapely_polygon.is_valid:
            return shapely_polygon

        fixed_shapely_polygon = shapely_polygon.buffer(0)

        if self.draw_invalid_polygons:
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
        x1pa, y1pa, x2pa, y2pa = self._get_processing_area_for_polygon(current_polygon)

        # self._save_tile((x1pa, y1pa, x2pa, y2pa), dir_path=SMALL_WITH_TUMOR_IMAGES_DIR, ext='tif')
        x = range(x1pa, x2pa, TILE_STEP)
        y = range(y1pa, y2pa, TILE_STEP)
        coords_to_extract = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

        if coords_to_extract.shape[0] > MAX_TILES_PER_TUMOR:
            return pd.DataFrame()

        print(self._get_bounding_box_for_polygon(current_polygon),
              self._get_processing_area_for_polygon(current_polygon))

        tile_boxes = np.append(coords_to_extract, coords_to_extract + TILE_SIZE, axis=1)

        columns = ['x1', 'y1', 'x2', 'y2']
        df = pd.DataFrame(tile_boxes, columns=columns)

        df['class_name'] = \
            df.apply(lambda tile_box: self._classify_tile_box(tile_box, global_multipolygon),
                     axis=1)

        with ThreadPoolExecutor() as executor:
            paths = executor.map(lambda tile_box: self._save_tile(tile_box),
                                 df.itertuples(index=False))

        df['path'] = pd.DataFrame(paths)

        return df

    def _classify_tile_box(self, tile_box, global_multipolygon):
        return DEFAULT_CLASS_NAME \
            if self._is_contained(global_multipolygon, tile_box=tile_box) else BACKGROUND_CLASS_NAME

    def _get_processing_area_for_polygon(self, polygon):
        x1, y1, x2, y2 = self._get_bounding_box_for_polygon(polygon)

        # enlarge_area_x = int(AREA_PROCESSING_MULTIPLIER * TILE_SIZE)
        # enlarge_area_y = int(AREA_PROCESSING_MULTIPLIER * TILE_SIZE)
        enlarge_area_x = int((x2 - x1) * (AREA_PROCESSING_MULTIPLIER - 1) / 2) + 1
        enlarge_area_y = int((y2 - y1) * (AREA_PROCESSING_MULTIPLIER - 1) / 2) + 1

        return max(x1 - enlarge_area_x, 0), \
               max(y1 - enlarge_area_y, 0), \
               min(x2 + enlarge_area_x, self.slide.dimensions[0]), \
               min(y2 + enlarge_area_y, self.slide.dimensions[1])

    def _is_contained(self, polygon, tile_box):
        rect = box(*tile_box)
        return polygon.intersects(scale(rect,
                                        AREA_TO_INTERSECT_MULTIPLIER, AREA_TO_INTERSECT_MULTIPLIER))

    def _calculate_local_bounding_boxes(self, bbox, tile_box):
        poly = box(*tile_box).intersection(bbox)
        x1, y1, x2, y2 = tile_box
        global_boxes = self._get_bounding_boxes_for_geometry(poly)
        local_boxes = [(x1g - x1, y1g - y1, x2g - x1, y2g - y1) for x1g, y1g, x2g, y2g in
                       global_boxes]
        return local_boxes

    def _get_bounding_box_for_polygon(self, poly):
        points = np.array(poly.exterior.coords.xy).astype(np.int)
        return points[0].min(), points[1].min(), points[0].max(), points[1].max()

    def _get_bounding_boxes_for_geometry(self, geo):
        try:
            geoms = [geo] if hasattr(geo, 'exterior') else \
                filter(lambda x: hasattr(x, 'exterior'), geo.geoms)

            return list(map(self._get_bounding_box_for_polygon, geoms))
        except:
            print('Unexpected type:', type(geo))
            return []

    def _save_tile(self, tile_box, ext='png'):
        tile = self.cut_tile(tile_box.x1, tile_box.y1, tile_box.x2 - tile_box.x1,
                             tile_box.y2 - tile_box.y1).resize(NETWORK_INPUT_SHAPE[:2])

        image_name = f"{basename(self.slide_path)}_({tile_box.x1}-{tile_box.y1}-{tile_box.x2}-{tile_box.y2}).{ext}"
        image_path = SOURCE_PATH / image_name

        tile.save(image_path)
        return image_path
