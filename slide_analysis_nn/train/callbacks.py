from glob import glob
from os import path, unlink

import keras
import tensorflow
from tensorflow.python.framework.graph_util import convert_variables_to_constants


class TB(keras.callbacks.TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue

                summary = tensorflow.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)

    def on_epoch_end(self, *args, **kwargs):
        self.on_batch_end(*args, **kwargs)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


class TensorGraphConverter(keras.callbacks.ModelCheckpoint):
    def __init__(self, model, filepath, **kwargs):
        super(TensorGraphConverter, self).__init__(filepath, **kwargs)
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        frozen_graph = self._freeze_session(keras.backend.get_session())
        tensorflow.train.write_graph(frozen_graph, self.filepath,
                                     'resnet50_{{epoch:02d}}.pb', as_text=False)

    def _freeze_session(self, session, keep_var_names=None, output_names=None,
                        clear_devices=True):
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(
                set(v.op.name for v in tensorflow.global_variables()).difference(
                    keep_var_names or []
                )
            )
            output_names = output_names or []
            output_names += [v.op.name for v in tensorflow.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                          output_names, freeze_var_names)
            return frozen_graph


class BestModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self._clean_folder_except_last_created()

    def _clean_folder_except_last_created(self):
        directory = path.dirname(self.filepath)
        mask = path.join(directory, '*.h5')
        files = glob(mask)
        files.sort(key=path.getmtime)
        for f in files[:-1]:
            unlink(f)
