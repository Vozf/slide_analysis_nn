import uuid

import cv2
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openslide import open_slide
import pandas as pd

from slide_analysis_nn.train.datasets_preparation.settings import (
    DEFAULT_CLASS_NAME,
    BACKGROUND_CLASS_NAME,
    TRAIN_DIR_NAME
)
from slide_analysis_nn.train.settings import (
    NUMBER_OF_SAMPLES_PER_SLIDE,
    HEALTHY_MASK_NUM_OF_DILLATIONS
)
from slide_analysis_nn.utils.ASAP_xml import append_polygons_to_existing_xml
from slide_analysis_nn.tile import TILE_SIZE
from slide_analysis_nn.utils.types import Area_box


class Slide:
    def __init__(self, slide_path: str, create_xml_with_cut_tiles: bool = False):
        self.slide_path, self.create_xml_with_cut_tiles = slide_path, create_xml_with_cut_tiles

        self.slide = open_slide(slide_path)

        self.log = logging.getLogger('slide')

        # default logging level, can be replaced by running --log=info
        logging.basicConfig()
        self.log.setLevel(logging.INFO)

    def cut_tile(self, x: int, y: int, width: int = TILE_SIZE, height: int = TILE_SIZE):
        return self.slide.read_region((x, y), 0, (width, height))

    def generate_df_from_mask(self, mask_path: str) -> pd.DataFrame:
        tumor_mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_RGB2GRAY)

        healthy_mask = self._get_healthy_mask(tumor_mask)

        tumor_df = self._produce_samples_from_mask(tumor_mask, NUMBER_OF_SAMPLES_PER_SLIDE // 2)
        healthy_df = self._produce_samples_from_mask(healthy_mask, NUMBER_OF_SAMPLES_PER_SLIDE // 2)

        tumor_df['class_name'] = DEFAULT_CLASS_NAME
        healthy_df['class_name'] = BACKGROUND_CLASS_NAME

        df = pd.concat((tumor_df, healthy_df))

        df['slide_path'] = self.slide_path

        if self.create_xml_with_cut_tiles:
            self._create_xml_annotation_with_marked_tiles(df)

        return df

    def _get_healthy_mask(self, tumor_mask: np.ndarray) -> np.ndarray:
        """
        Predicting non tumor on tumor mask gets mostly background.
        So we create another mask which is essentially the surrounding area of tumor.
        In this area tiles aren't background. They are healthy cells.
        """
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(tumor_mask, kernel, iterations=HEALTHY_MASK_NUM_OF_DILLATIONS)
        tumor_border = cv2.dilate(tumor_mask, kernel, iterations=1)
        return dilation - tumor_border

    def _produce_samples_from_mask(self, mask: np.ndarray, num_samples: int) -> pd.DataFrame:
        tile_boxes = self._get_tile_boxes(mask, num_samples)

        columns = ['x1', 'y1', 'x2', 'y2']
        df = pd.DataFrame(tile_boxes, columns=columns)

        with ThreadPoolExecutor() as executor:
            paths = executor.map(self._save_tile, df.itertuples(index=False))

        df['path'] = pd.DataFrame(paths)
        return df

    def _get_tile_boxes(self, mask: np.ndarray, num_samples: int) -> np.ndarray:
        mask_coords = np.flip(np.column_stack(np.nonzero(mask == 1)), axis=1)

        if len(mask_coords) < num_samples:
            self.log.warning(f'Not enough samples ({len(mask_coords)})'
                             f' in {os.path.basename(self.slide_path)}.'
                             f' Upsampling to {num_samples}')

            mask_coords = np.repeat(mask_coords, num_samples / len(mask_coords) + 1, axis=0)

        np.random.shuffle(mask_coords)
        mask_coords = mask_coords[:num_samples]

        MASK_DOWNSCALE_FACTOR = 32
        slide_coords_of_center = np.asarray(
            [(coord * MASK_DOWNSCALE_FACTOR + MASK_DOWNSCALE_FACTOR / 2) for coord in mask_coords])

        slide_coords_of_upleft = np.asarray(
            [(center - TILE_SIZE // 2).astype(int) for center in slide_coords_of_center])

        return np.column_stack((slide_coords_of_upleft, slide_coords_of_upleft + TILE_SIZE))

    def _create_xml_annotation_with_marked_tiles(self, df: pd.DataFrame):
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

    def _save_tile(self, tile_box: Area_box, ext: str = 'png') -> str:
        tile = self.cut_tile(tile_box.x1, tile_box.y1, tile_box.x2 - tile_box.x1,
                             tile_box.y2 - tile_box.y1)

        image_name = f"{os.path.basename(self.slide_path)}" \
                     f"_({tile_box.x1}-{tile_box.y1}-{tile_box.x2}-{tile_box.y2})" \
                     f"+{uuid.uuid4().hex[:6]}.{ext}"
        image_path = TRAIN_DIR_NAME / image_name

        tile.save(image_path)
        return str(image_path)
