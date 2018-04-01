import csv
import glob
import itertools
import os
import random
from functools import reduce

import time
from collections import defaultdict
from numbers import Integral
import logging
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

from utils.ASAP_xml import read_polygons_xml
from utils.slide import Slide

from train.datasets_preparation.settings import (
    LABELED_IMAGES_DIR,
    AUGMENTATION_PERCENT,
    TRAIN_DATASET_FILE_PATH,
    TEST_DATASET_FILE_PATH,
    TRAIN_DATASET_PERCENT,
    AUGMENTED_IMAGES_DIR,
    UNLABELED_IMAGES_DIR,
    DEFAULT_CLASS_NAME,
    CLASS_MAPPING_FILE_PATH,
    SLIDE_IMAGES_DIR
)
from train.datasets_preparation.utils import Label


class DatasetPreparation(object):
    def __init__(self):
        self.log = logging.getLogger('datasets.preparation')
        self.labeled_images = defaultdict(list)
        self.unlabeled_images = defaultdict(list)
        self.augmented_images = defaultdict(list)

    def _prepare_slides_for_training(self):
        for the_file in os.listdir(LABELED_IMAGES_DIR):
            file_path = os.path.join(LABELED_IMAGES_DIR, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        for the_file in os.listdir(UNLABELED_IMAGES_DIR):
            file_path = os.path.join(UNLABELED_IMAGES_DIR, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        polygon_images = map(self._prepare_polygons,
                             glob.iglob(os.path.join(SLIDE_IMAGES_DIR, '*xml')))
        start = time.time()

        dicts = list(map(self._process_slide, filter(lambda x: x, polygon_images)))

        print('cut')
        print(time.time() - start)
        return dicts

    def _unite_array_of_dictionaries(self, list_of_dict_pairs):
        return reduce(lambda acc, dic: ({**dic[0], **dic[1], **acc}),
                      list_of_dict_pairs, {})


    def _prepare_polygons(self, xml_file_path):
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
        class_names = {label.class_name for labels in datasets_values for label in labels if label.class_name}

        return [list(reversed(i)) for i in enumerate(class_names)]

    def augment_images(self, images_to_augment, batch_size=50):
        augmentator = iaa.OneOf(
            [
                iaa.GaussianBlur((0.2, 1)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.02, 0.05), per_channel=0.5),
                iaa.Add((-30, 30)),
                iaa.Affine(
                    translate_percent={
                        "x": (-0.03, 0.03),
                        "y": (-0.03, 0.03)
                    },
                    mode='edge'
                ),
                iaa.Affine(
                    rotate=(-15, 15),
                    mode='edge'
                ),
                iaa.Affine(
                    shear=(-10, 10),
                    mode='edge'
                ),
            ],
        )

        file_names = list(images_to_augment.keys())

        for i in range(0, len(file_names), batch_size):
            keypoints_on_images = []
            images = []
            batch_file_names = file_names[i: i + batch_size]
            for file_name in batch_file_names:
                image = cv2.imread(file_name)
                images.append(image)

                bboxes = images_to_augment[file_name]

                keypoints = []
                for bbox in bboxes:
                    keypoints.append(ia.Keypoint(x=int(bbox.x1), y=int(bbox.y1)))
                    keypoints.append(ia.Keypoint(x=int(bbox.x2), y=int(bbox.y1)))
                    keypoints.append(ia.Keypoint(x=int(bbox.x2), y=int(bbox.y2)))
                    keypoints.append(ia.Keypoint(x=int(bbox.x1), y=int(bbox.y2)))

                keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))

            aug_det = augmentator.to_deterministic()

            images_aug = aug_det.augment_images(images)
            keypoints_aug = aug_det.augment_keypoints(keypoints_on_images)

            for file_name, augmented_image, image_aug_keypoints in zip(batch_file_names, images_aug, keypoints_aug):
                basename = os.path.basename(file_name)
                augmented_image_path = os.path.join(AUGMENTED_IMAGES_DIR, basename)
                for j in range(0, len(image_aug_keypoints.keypoints), 4):
                    p1, p2, p3, p4 = image_aug_keypoints.keypoints[j: j + 4]

                    self.augmented_images[augmented_image_path].append(
                        Label(
                            path=augmented_image_path,
                            x1=min([p1.x, p2.x, p3.x, p4.x]),
                            y1=min([p1.y, p2.y, p3.y, p4.y]),
                            x2=max([p1.x, p2.x, p3.x, p4.x]),
                            y2=max([p1.y, p2.y, p3.y, p4.y]),
                            class_name=images_to_augment[file_name][j // 4].class_name
                        )
                    )


                cv2.imwrite(augmented_image_path, augmented_image)

        return self.augmented_images

    def populate_prepared_datasets(self):
        test_finished = defaultdict(list)
        train_finished = defaultdict(list)
        # images_to_augment = defaultdict(list)

        dicts = self._prepare_slides_for_training()

        train_prepared, val_prepared = self._divide_to_train_and_validation(dicts)

        # number_of_images_to_augment = int(AUGMENTATION_PERCENT * number_of_train_tumors)

        # get images from train set to augmentation
        # images_to_augment.update(
        #     {
        #         i: train_prepared[i] for i in list(train_prepared.keys())[:number_of_images_to_augment]
        #     }
        # )
        # augmented_images = self.augment_images(images_to_augment)

        # add augmented and unlabeled images to train set, shuffle train set
        # train_prepared.update(
        #     {
        #         i: self.unlabeled_images[i] for i in augmented_images
        #     }
        # )
        # train_prepared.update(self.augmented_images)

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
        self.create_csv_from_list(
            [label for labels in test_finished.values() for label in labels],
            TEST_DATASET_FILE_PATH
        )
        self.create_csv_from_list(
            [label for labels in train_finished.values() for label in labels],
            TRAIN_DATASET_FILE_PATH
        )

    def _divide_to_train_and_validation(self, dicts):
        random.shuffle(dicts)

        dataset_size = sum([len(x) for x in dicts])
        ideal_number_of_train_tumors = int(dataset_size * TRAIN_DATASET_PERCENT)
        current_number_of_train_tumors = 0

        train = defaultdict(list)
        test = defaultdict(list)

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

            csv_data = [
                ','.join(
                    map(lambda x: DatasetPreparation.make_string(x), tuple(line))
                ) for line in data
            ]

            list_writer.writerow(csv_data)

    @staticmethod
    def make_string(item):
        return str(item) if isinstance(item, Integral) else '' if item is None else item

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
