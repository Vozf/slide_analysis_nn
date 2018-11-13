import csv
import glob
import logging
import os
import random
import time
import pandas as pd

from slide_analysis_nn.train.datasets_preparation.settings import (
    TRAIN_DATASET_FILE_PATH,
    TEST_DATASET_FILE_PATH,
    CLASS_MAPPING_FILE_PATH,
    SLIDE_IMAGES_DIR,
    FULL_DATASET_FILE_PATH,
    TEST_DIR_NAME,
    TRAIN_DIR_NAME,
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

    def create_dataset(self):
        df = self._prepare_slides_for_training()

        uniques = self._encode_class_name(df)
        self.create_class_mapping_csv(uniques)

        self._save_train_test_split(df)

        if self.save_full_dataset_csv:
            df.to_csv(FULL_DATASET_FILE_PATH, index=False)

    # def generate_new_train_test_split_from_full_dataset(self, full_csv_path=FULL_DATASET_FILE_PATH):
    #     full_df = pd.read_csv(full_csv_path)
    #
    #     self._save_train_test_split(full_df)

    def _save_train_test_split(self, full_df):
        train, test = self._train_test_split(full_df)

        self._move_samples_to_folders(train, TRAIN_DIR_NAME)
        self._move_samples_to_folders(test, TEST_DIR_NAME)

        self._print_statistics(train, 'Train')
        self._print_statistics(test, 'Test')

        train[['path', 'class_encoded']].to_csv(TRAIN_DATASET_FILE_PATH, index=False)
        test[['path', 'class_encoded']].to_csv(TEST_DATASET_FILE_PATH, index=False)

    def _move_samples_to_folders(self, df, folder):
        if df.empty:
            return

        def move_to_folder(row):
            destination = folder / row.class_name / os.path.basename(row.path)
            os.rename(row.path, destination)
            return str(destination)

        df['path'] = df.apply(move_to_folder, axis=1)

    def _encode_class_name(self, df):
        labels, uniques = df.class_name.factorize(sort=True)
        df['class_encoded'] = labels
        df.class_encoded = df.class_encoded.astype(int)
        return uniques

    def _prepare_slides_for_training(self):
        for file in glob.glob(str(TRAIN_DIR_NAME / '*' / '*')):
            os.remove(file)
        for file in glob.glob(str(TEST_DIR_NAME / '*' / '*')):
            os.remove(file)

        xmls = glob.iglob(str(SLIDE_IMAGES_DIR / '*xml'))

        polygon_images = list(filter(None, map(self._get_polygons_from_xml, xmls)))
        # polygon_images = list(filter(None, map(self._get_polygons_from_xml, xmls)))[:4]

        start = time.time()

        dfs = []
        for polygon_image in polygon_images:
            try:
                dfs.append(self._process_slide(polygon_image))
            except Exception as e:
                print(e)
                continue

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

    def _print_statistics(self, df, df_name='Dataset'):
        df_class_distribution = df.class_name.value_counts(normalize=True)
        df_slide_tiles_percentage = df.slide_path.str[-13:].value_counts(normalize=True)
        self.log.info('-' * 50)
        self.log.info(f'{df_name} data distribution of {len(df)} samples:')
        self.log.info(df_class_distribution.to_string())
        self.log.info('')
        self.log.info(df_slide_tiles_percentage.to_string())

    @staticmethod
    def _train_test_split(df):
        num_samples_df = df.groupby(['slide_path']).size()
        num_samples_in_slide = list(zip(num_samples_df.keys().values, num_samples_df.values))
        random.shuffle(num_samples_in_slide)

        ideal_number_of_train_tumors = int(len(df) * TRAIN_TEST_DATASET_PERCENT)
        current_train_num_samples = 0

        train = pd.DataFrame(columns=df.columns)
        test = pd.DataFrame(columns=df.columns)

        for slide_path, num_samples in num_samples_in_slide:
            if current_train_num_samples < ideal_number_of_train_tumors:
                train = train.append(df.where(df.slide_path == slide_path).dropna())
                current_train_num_samples += num_samples
            else:
                test = test.append(df.where(df.slide_path == slide_path).dropna())

        print('Train data set percentage = {:.2%}'.format(
            current_train_num_samples / len(df)))

        train.class_encoded = train.class_encoded.astype(int)
        test.class_encoded = test.class_encoded.astype(int)
        return train, test

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
