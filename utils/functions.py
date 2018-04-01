from functools import reduce


def dict_assign(*args):
    return reduce(lambda acc, dic: {**acc, **dic}, args)
