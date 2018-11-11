import csv
import glob
import logging
import os
import random
import time
import pandas as pd

from slide_analysis_nn.train.datasets_preparation.settings import (
    LABELED_IMAGES_DIR,
    TRAIN_DATASET_FILE_PATH,
    TEST_DATASET_FILE_PATH,
    UNLABELED_IMAGES_DIR,
    CLASS_MAPPING_FILE_PATH,
    SLIDE_IMAGES_DIR
)
from slide_analysis_nn.train.settings import TRAIN_TEST_DATASET_PERCENT
from slide_analysis_nn.utils.ASAP_xml import read_polygons_xml
from slide_analysis_nn.utils.slide import Slide


class DatasetPreparation(object):
    def __init__(self, save_full_dataset_csv=True):
        self.save_full_dataset_csv = save_full_dataset_csv
        self.log = logging.getLogger('datasets.preparation')

        # default logging level, can be replaced by running --log=info
        logging.basicConfig()
        self.log.setLevel(logging.INFO)

    def _prepare_slides_for_training(self):
        for the_file in os.listdir(LABELED_IMAGES_DIR):
            file_path = LABELED_IMAGES_DIR / the_file
            if os.path.isfile(file_path):
                os.unlink(file_path)

        for the_file in os.listdir(UNLABELED_IMAGES_DIR):
            file_path = UNLABELED_IMAGES_DIR / the_file
            if os.path.isfile(file_path):
                os.unlink(file_path)

        xmls = glob.iglob(str(SLIDE_IMAGES_DIR / '*xml'))

        polygon_images = list(filter(None, map(self._get_polygons_from_xml, xmls)))

        start = time.time()

        dfs = map(self._process_slide, polygon_images)

        df = pd.concat(filter(lambda df: not df.empty, dfs))

        print('cut')
        print(time.time() - start)

        return df

    def _get_polygons_from_xml(self, xml_file_path):
        image_path = '{}.tif'.format(os.path.splitext(xml_file_path)[0])
        if not os.path.exists(image_path):
            self.log.warning("Path {0} not exist".format(image_path))
            return None

        polygons = read_polygons_xml(xml_file_path)

        return {
            'image_path': image_path,
            'polygons': polygons,
        }

    def _process_slide(self, polygon_images):
        slide = Slide(polygon_images['image_path'])

        return slide.cut_polygons_data(polygon_images['polygons'])

    def populate_prepared_datasets(self):
        df = self._prepare_slides_for_training()

        labels, uniques = df.class_name.factorize(sort=True)
        df['class_encoded'] = labels
        df.class_encoded = df.class_encoded.astype(int)
        self.create_class_mapping_csv(uniques)

        train, val = self._divide_to_train_and_validation(df)

        self._print_statistics(train, 'Train')
        self._print_statistics(val, 'Test')

        if self.save_full_dataset_csv:
            df.to_csv(TRAIN_DATASET_FILE_PATH.parent / 'full.csv', index=False)

        train[['path', 'class_encoded']].to_csv(TRAIN_DATASET_FILE_PATH, index=False)
        val[['path', 'class_encoded']].to_csv(TEST_DATASET_FILE_PATH, index=False)

    def _print_statistics(self, df, df_name='Dataset'):
        df_class_distribution = df.class_name.value_counts(normalize=True)
        df_slide_tiles_percentage = df.slide_path.str[-13:].value_counts(normalize=True)
        self.log.info('-' * 8)
        self.log.info(f'{df_name} data distribution:')
        self.log.info(df_class_distribution.to_string())
        self.log.info('')
        self.log.info(df_slide_tiles_percentage.to_string())

    @staticmethod
    def _divide_to_train_and_validation(df):
        num_samples_df = df.groupby(['slide_path']).size()
        num_samples_in_slide = list(zip(num_samples_df.keys().values, num_samples_df.values))
        random.shuffle(num_samples_in_slide)

        ideal_number_of_train_tumors = int(len(df) * TRAIN_TEST_DATASET_PERCENT)
        current_train_num_samples = 0

        train = pd.DataFrame(columns=df.columns)
        val = pd.DataFrame(columns=df.columns)


        for slide_path, num_samples in num_samples_in_slide:
            if current_train_num_samples < ideal_number_of_train_tumors:
                train = train.append(df.where(df.slide_path == slide_path).dropna())
                current_train_num_samples += num_samples
            else:
                val = val.append(df.where(df.slide_path == slide_path).dropna())

        print('Train data set percentage = {:.2%}'.format(
            current_train_num_samples / len(df)))

        train.class_encoded = train.class_encoded.astype(int)
        val.class_encoded = val.class_encoded.astype(int)
        return train, val

    @staticmethod
    def create_class_mapping_csv(uniques):
        pd.DataFrame(list(enumerate(uniques))).reindex(columns=[1, 0])\
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
