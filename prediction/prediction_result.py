import os

import numpy as np

from train.datasets_preparation.settings import DEFAULT_CLASS_NAME
from utils.ASAP_xml import write_polygons_xml
from utils.constants import TILE_SIZE


class PredictionResult:
    def __init__(self, predict_generator, predicted_labels, scores):
        self.predict_generator = predict_generator
        self.slide_path = predict_generator.slide.slide_path
        self.predicted_labels = predicted_labels
        self.scores = scores

    def create_asap_annotations(self):
        xml_path = '{}_predicted.xml'.format(os.path.splitext(self.slide_path)[0])
        polygons = np.asarray([
            [(x1, y1), (x1 + TILE_SIZE, y1), (x1 + TILE_SIZE, y1 + TILE_SIZE), (x1, y1 + TILE_SIZE)]
            for (x1, y1) in self.predict_generator.addresses])

        chosen_idx = self.predicted_labels == self.predict_generator.label_names_to_id[DEFAULT_CLASS_NAME]

        return write_polygons_xml(polygons[chosen_idx],
                                  predicted_labels=self.predicted_labels[chosen_idx],
                                  scores=self.scores[chosen_idx], xml_path=xml_path)
