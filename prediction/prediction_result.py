import os

import numpy as np

from prediction.settings import DEFAULT_SCORE_THRESHOLD
from train.datasets_preparation import DatasetPreparation
from train.datasets_preparation.settings import DEFAULT_CLASS_NAME
from utils.ASAP_xml import write_polygons_xml, append_polygons_to_existing_xml
from utils.constants import TILE_SIZE


class PredictionResult:
    def __init__(self, image_path, predicted_labels, scores, tile_coordinates=None):
        self.image_path = image_path
        self.predicted_labels = predicted_labels
        self.scores = scores
        self.tile_coordinates = tile_coordinates
        self.label_name_to_label_id = DatasetPreparation.get_label_name_to_label_id_dict()
        self.label_id_to_label_name = DatasetPreparation.get_label_id_to_label_name_dict()

    def save_as_asap_annotations(self, score_threshold=DEFAULT_SCORE_THRESHOLD,
                                 truth_xml_path=None):
        xml_path = '{}_predicted.xml'.format(os.path.splitext(self.image_path)[0])
        polygons = np.asarray([
            [(x1, y1), (x1 + TILE_SIZE, y1), (x1 + TILE_SIZE, y1 + TILE_SIZE), (x1, y1 + TILE_SIZE)]
            for (x1, y1) in self.tile_coordinates])

        chosen_idx = (self.predicted_labels == self.label_name_to_label_id[
            DEFAULT_CLASS_NAME]) & (self.scores > score_threshold)

        if truth_xml_path:
            append_polygons_to_existing_xml(polygons[chosen_idx],
                                            predicted_labels=self.predicted_labels[chosen_idx],
                                            scores=self.scores[chosen_idx],
                                            source_xml_path=truth_xml_path,
                                            output_xml_path=xml_path)
        else:
            return write_polygons_xml(polygons[chosen_idx],
                                      predicted_labels=self.predicted_labels[chosen_idx],
                                      scores=self.scores[chosen_idx], xml_path=xml_path)


    def __str__(self):
        return 'Path: {0}\nscores: {1}\nlabels: {2}'.format(self.image_path, self.scores, list(map(
            lambda x: self.label_id_to_label_name[x], self.predicted_labels)))
