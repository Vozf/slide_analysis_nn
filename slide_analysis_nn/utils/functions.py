from functools import reduce
import requests


def dict_assign(*args):
    return reduce(lambda acc, dic: {**acc, **dic}, args)


def download_file(url, file_name):
    file_name = file_name if file_name else url.split('/')[-1]

    r = requests.get(url, stream=True)
    with open(file_name, 'wb+') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return file_name