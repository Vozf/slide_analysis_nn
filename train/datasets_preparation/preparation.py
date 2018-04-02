import csv
import glob
import itertools
import os
import random
import time
import logging

from utils.ASAP_xml import read_polygons_xml
from utils.slide import Slide

from train.datasets_preparation.settings import (
    LABELED_IMAGES_DIR,
    TRAIN_DATASET_FILE_PATH,
    TEST_DATASET_FILE_PATH,
    TRAIN_DATASET_PERCENT,
    UNLABELED_IMAGES_DIR,
    CLASS_MAPPING_FILE_PATH,
    SLIDE_IMAGES_DIR
)


class DatasetPreparation(object):
    def __init__(self):
        self.log = logging.getLogger('datasets.preparation')

    def _prepare_slides_for_training(self):
        for the_file in os.listdir(LABELED_IMAGES_DIR):
            file_path = os.path.join(LABELED_IMAGES_DIR, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        for the_file in os.listdir(UNLABELED_IMAGES_DIR):
            file_path = os.path.join(UNLABELED_IMAGES_DIR, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        xmls = glob.iglob(os.path.join(SLIDE_IMAGES_DIR, '*xml'))

        polygon_images = map(self._get_polygons_from_xml, xmls)

        start = time.time()

        dicts = list(map(self._process_slide, filter(None, polygon_images)))

        print('cut')
        print(time.time() - start)

        return dicts

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

    def get_class_mapping(self, train_dataset, test_dataset):
        datasets_values = itertools.chain(train_dataset.values(), test_dataset.values())
        class_names = {label for label in datasets_values}

        return [list(reversed(i)) for i in enumerate(class_names)]

    def populate_prepared_datasets(self):
        test_finished = {}
        train_finished = {}

        dicts = self._prepare_slides_for_training()

        train_prepared, val_prepared = self._divide_to_train_and_validation(dicts)

        shuffled_train = list(train_prepared.keys())
        random.shuffle(shuffled_train)
        train_finished.update(
            {
                i: train_prepared[i] for i in shuffled_train
            }
        )

        shuffled_test = list(val_prepared.keys())
        random.shuffle(shuffled_test)
        test_finished.update(
            {
                i: val_prepared[i] for i in shuffled_test
            }
        )

        class_mapping = self.get_class_mapping(test_finished, train_finished)

        self.create_csv_from_list(class_mapping, CLASS_MAPPING_FILE_PATH)

        # create CSV files from train and test sets
        self.create_csv_from_list(test_finished.items(), TEST_DATASET_FILE_PATH)
        self.create_csv_from_list(train_finished.items(), TRAIN_DATASET_FILE_PATH)

    @staticmethod
    def _divide_to_train_and_validation(dicts):
        random.shuffle(dicts)

        dataset_size = sum([len(x) for x in dicts])
        ideal_number_of_train_tumors = int(dataset_size * TRAIN_DATASET_PERCENT)
        current_number_of_train_tumors = 0

        train = {}
        test = {}

        for tumor_data in dicts:
            if current_number_of_train_tumors < ideal_number_of_train_tumors:
                train.update(tumor_data)
                current_number_of_train_tumors += len(tumor_data)
            else:
                test.update(tumor_data)

        print('Train data set percentage = {:.2%}'.format(
            current_number_of_train_tumors / dataset_size))

        return train, test

    @staticmethod
    def create_csv_from_list(data, csv_path):
        with open(csv_path, 'w', newline='') as csv_file:
            list_writer = csv.writer(csv_file, delimiter='\n')

            csv_data = [','.join(map(str, line)) for line in data]

            list_writer.writerow(csv_data)

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
