import os

import numpy as np
from PIL import Image
from matplotlib import cm

from slide_analysis_nn.prediction.settings import DEFAULT_SCORE_THRESHOLD
from slide_analysis_nn.train.datasets_preparation import DatasetPreparation
from slide_analysis_nn.train.datasets_preparation.settings import DEFAULT_CLASS_NAME
from slide_analysis_nn.utils.ASAP_xml import write_polygons_xml, append_polygons_to_existing_xml
from slide_analysis_nn.tile import TILE_SIZE, TILE_STEP


class PredictionResult:
    def __init__(self, image_path, scores, tile_coordinates=None):
        self.image_path = image_path
        self.all_scores = scores

        self.predicted_labels = np.argmax(self.all_scores, axis=1)
        self.predicted_labels_scores = self.all_scores[
            np.arange(self.all_scores.shape[0]), self.predicted_labels]

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
            DEFAULT_CLASS_NAME]) & (self.predicted_labels_scores > score_threshold)

        if truth_xml_path:
            return append_polygons_to_existing_xml(polygons[chosen_idx],
                                                   predicted_labels=self.predicted_labels[
                                                       chosen_idx],
                                                   scores=self.predicted_labels_scores[chosen_idx],
                                                   source_xml_path=truth_xml_path,
                                                   output_xml_path=xml_path)
        else:
            return write_polygons_xml(polygons[chosen_idx],
                                      predicted_labels=self.predicted_labels[chosen_idx],
                                      scores=self.predicted_labels_scores[chosen_idx],
                                      xml_path=xml_path)

    def __str__(self):
        return 'Path: {0}\nscores: {1}\nlabels: {2}'.format(self.image_path,
                                                            self.predicted_labels_scores, list(map(
                lambda x: self.label_id_to_label_name[x], self.predicted_labels)))

    def create_map(self, path=None):
        image_size = (
                (np.max(self.tile_coordinates[:, 0]) - np.min(
                    self.tile_coordinates[:, 0])) // TILE_STEP + 1,
                (np.max(self.tile_coordinates[:, 1]) - np.min(
                    self.tile_coordinates[:, 1])) // TILE_STEP + 1
        )
        img_base_name = os.path.splitext(self.image_path)[0]

        map = cm.ScalarMappable(cmap='jet')
        tumor_scores = self.get_tumor_scores()

        resized = np.reshape(tumor_scores, image_size[::-1])
        rgba = map.to_rgba(resized, bytes=True)

        path = path if path else img_base_name + '_color_map.png'

        return Image.fromarray(rgba).save(path)

    def get_tumor_scores(self):
        return self.all_scores[:, self.label_name_to_label_id[DEFAULT_CLASS_NAME]]

    def get_results(self):
        return [{
            "x": int(coord[0]),
            "y": int(coord[1]),
            "height": TILE_SIZE,
            "width": TILE_SIZE,
            "score": float(score),
        } for coord, score in zip(self.tile_coordinates, self.get_tumor_scores())]
