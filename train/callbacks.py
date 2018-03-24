from os import path, unlink
from glob import glob
import keras
import tensorflow
from tensorflow.python.framework.graph_util import convert_variables_to_constants


class TensorBoardRetinaCallback(keras.callbacks.TensorBoard):
    def __init__(self, write_batch_performance, *args, **kwargs):
        super(TensorBoardRetinaCallback, self).__init__(*args, **kwargs)

        self.write_batch_performance = write_batch_performance
        self.seen = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if not self.validation_data and self.histogram_freq:
            raise ValueError('If printing histograms, validation_data must be provided, and cannot be a generator.')

        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [keras.backend.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, self.seen)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tensorflow.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.seen)
        self.writer.flush()
        self.seen += self.batch_size

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.write_batch_performance:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tensorflow.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.seen)
            self.writer.flush()

        self.seen += self.batch_size


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
