from functools import reduce


def dict_assign(*args):
    reduce(lambda acc, dic: {**acc, **dic}, args)