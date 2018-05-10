import os
import re
import subprocess


class GPUSupportMixin(object):
    def set_up_gpu(self, gpu_ids):
        if gpu_ids:
            self.log.warning('Picking GPU {}'.format(gpu_ids))
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '\n'.join(gpu_ids)
        else:
            self.log.warning('GPU not found')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def _get_available_gpus(self):
        """Returns list of available GPU ids."""
        output = subprocess.Popen(
            "nvidia-smi -L", stdout=subprocess.PIPE, shell=True
        ).communicate()[0].decode("ascii")

        # lines of the form GPU 0: TITAN X
        gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
        gpu_ids = []
        for line in filter(lambda x: bool(x), output.strip().split("\n")):
            m = gpu_regex.match(line)
            gpu_ids.append(m.group("gpu_id"))

        return gpu_ids
