import math
from functools import reduce
import requests
from tqdm import tqdm


def dict_assign(*args):
    return reduce(lambda acc, dic: {**acc, **dic}, args)


def download_file(url, file_name):
    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    with open(file_name, 'wb') as f:
        for chunk in tqdm(r.iter_content(block_size), total=math.ceil(total_size / block_size),
                          unit='KB', unit_scale=True):
            f.write(chunk)
