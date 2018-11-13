import csv
import glob
import logging
import os
import random
import re
import time
import numpy as np
from pathlib import Path
from typing import Tuple

import pandas as pd

from slide_analysis_nn.train.datasets_preparation.settings import (
    TRAIN_DATASET_FILE_PATH,
    TEST_DATASET_FILE_PATH,
    CLASS_MAPPING_FILE_PATH,
    SLIDE_IMAGES_DIR,
    FULL_DATASET_FILE_PATH,
    TEST_DIR_NAME,
    TRAIN_DIR_NAME,
    LABELS,
)
from slide_analysis_nn.train.settings import TRAIN_TEST_DATASET_PERCENT
from slide_analysis_nn.utils.slide import Slide


class DatasetPreparation:
    def __init__(self, save_full_dataset_csv: bool = True):
        self.save_full_dataset_csv = save_full_dataset_csv
        self.log = logging.getLogger('datasets.preparation')

        # default logging level, can be replaced by running --log=info
        logging.basicConfig()
        self.log.setLevel(logging.INFO)

    def create_dataset(self):
        self._create_and_clean_data_folder(TRAIN_DIR_NAME)
        self._create_and_clean_data_folder(TEST_DIR_NAME)

        df = self._generate_dataset_df()

        uniques = self._encode_class_name(df)
        self.create_class_mapping_csv(uniques)

        dfs_with_changed_paths = self._save_train_test_split(df)

        df = pd.concat(dfs_with_changed_paths)

        if self.save_full_dataset_csv:
            df.to_csv(FULL_DATASET_FILE_PATH, index=False)

    @staticmethod
    def _create_and_clean_data_folder(folder_path: Path):
        [os.makedirs(folder_path / label, exist_ok=True) for label in LABELS]
        [os.remove(file) for file in
         glob.glob(str(folder_path / '**' / '*'), recursive=True) if os.path.isfile(file)]

    def generate_new_train_test_split_from_full_dataset(self,
                                                        full_csv_path: str = FULL_DATASET_FILE_PATH):
        full_df = pd.read_csv(full_csv_path)

        dfs_with_changed_paths = self._save_train_test_split(full_df)
        full_df = pd.concat(dfs_with_changed_paths)

        full_df.to_csv(FULL_DATASET_FILE_PATH, index=False)

    def _save_train_test_split(self, full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train, test = self._train_test_split(full_df)

        self._move_samples_to_folders(train, TRAIN_DIR_NAME)
        self._move_samples_to_folders(test, TEST_DIR_NAME)

        self._print_statistics(train, 'Train')
        self._print_statistics(test, 'Test')

        train[['path', 'class_encoded']].to_csv(TRAIN_DATASET_FILE_PATH, index=False)
        test[['path', 'class_encoded']].to_csv(TEST_DATASET_FILE_PATH, index=False)

        return train, test

    @staticmethod
    def _move_samples_to_folders(df: pd.DataFrame, folder: Path):
        if df.empty:
            return

        def move_to_folder(row):
            destination = folder / row.class_name / os.path.basename(row.path)
            os.rename(row.path, destination)
            return str(destination)

        df['path'] = df.apply(move_to_folder, axis=1)

    def _encode_class_name(self, df: pd.DataFrame) -> np.ndarray:
        labels, uniques = df.class_name.factorize(sort=True)
        df['class_encoded'] = labels
        df.class_encoded = df.class_encoded.astype(int)
        return uniques

    def _generate_dataset_df(self) -> pd.DataFrame:

        mask_paths = glob.glob(str(SLIDE_IMAGES_DIR / '*_evaluation_mask.png'))

        def get_slide_and_mask_path(mask_path):
            slide_path_no_ext = re.search('(.+)_evaluation_mask', mask_path).group(1)
            slide_path = f'{slide_path_no_ext}.tif'
            if not os.path.isfile(slide_path):
                self.log.warning(f'Path {slide_path} not exist')
                return None
            return slide_path, mask_path

        slide_mask_paths = filter(None, map(get_slide_and_mask_path, mask_paths))

        start = time.time()

        dfs = map(self._process_slide, slide_mask_paths)

        df = pd.concat(dfs)

        self.log.info(f'Slides cut in {time.time() - start} sec')

        return df

    def _process_slide(self, slide_and_mask_paths: Tuple[str, str]) -> pd.DataFrame:
        slide_path, mask_path = slide_and_mask_paths
        slide = Slide(slide_path)

        return slide.generate_df_from_mask(mask_path)

    def _print_statistics(self, df: pd.DataFrame, df_name: str = 'Dataset'):
        df_class_distribution = df.class_name.value_counts(normalize=True)
        df_slide_tiles_percentage = df.slide_path.str[-13:].value_counts(normalize=True)
        self.log.info('-' * 50)
        self.log.info(f'{df_name} data distribution of {len(df)} samples:')
        self.log.info(df_class_distribution.to_string())
        self.log.info('')
        self.log.info(df_slide_tiles_percentage.to_string())

    @staticmethod
    def _train_test_split(df: pd.DataFrame):
        unique_slides = df.slide_path.drop_duplicates().values
        random.shuffle(unique_slides)

        num_train_slides = round(TRAIN_TEST_DATASET_PERCENT * len(unique_slides))

        train_slides = unique_slides[:num_train_slides]
        test_slides = unique_slides[num_train_slides:]

        train = df.loc[df['slide_path'].isin(train_slides)].copy()
        test = df.loc[df['slide_path'].isin(test_slides)].copy()

        print('Train data set percentage = {:.2%}'.format(
            len(train) / len(df)))

        return train, test

    @staticmethod
    def create_class_mapping_csv(uniques: np.ndarray):
        pd.DataFrame(list(enumerate(uniques))).reindex(columns=[1, 0]) \
            .to_csv(CLASS_MAPPING_FILE_PATH, header=False, index=False)

    @staticmethod
    def get_label_name_to_label_id_dict():
        with open(CLASS_MAPPING_FILE_PATH, 'r') as csvfile:
            reader = csv.reader(csvfile)
            names_to_label_str = dict(reader)

        names_to_label_int = {k: int(v) for k, v in names_to_label_str.items()}

        return names_to_label_int

    @staticmethod
    def get_label_id_to_label_name_dict():
        name_to_id = DatasetPreparation.get_label_name_to_label_id_dict()
        id_to_name = {v: k for k, v in name_to_id.items()}

        return id_to_name
