from easydict import EasyDict as edict
import yaml
from utils.flatwhite import *
import torch
import numpy as np
import random
import platform

def easy_dic(dic):
    dic = edict(dic)
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = edict(value)
    return dic


def show_config(config, sub=False):
    msg = ''
    for key, value in config.items():
        if isinstance(value, dict):
            continue
        else:
            msg += '{:>25} : {:<15}\n'.format(key, value)
    return msg


def type_align(source, target):
    if isinstance(source, int):
        return int(target)
    elif isinstance(source, float):
        return float(target)
    elif isinstance(source, str):
        return target
    elif isinstance(source, bool):
        return bool(source)
    else:
        print("Unsupported type: {}".format(type(source)))


def config_parser(config, args):
    print(args)
    for arg in args:
        if '=' not in arg:
            continue
        else:
            key, value = arg.split('=')
        value = type_align(config[key], value)
        config[key] = value
    return config


def init_config(config_path, argvs):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    config = easy_dic(config)
    config = config_parser(config, argvs)
    config.snapshot = osp.join(config.snapshot, config.note)
    mkdir(config.snapshot)
    print('Snapshot stored in: {}'.format(config.snapshot))

    if config.fix_seed:
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
    message = show_config(config)
    print(message)
    return config


def set_exp_name(config):
    exp_name = '{}'.format(config.note)
    exp_name += '_D-{}'.format(config.source)
    exp_name += '_tar-{}'.format(config.target)
    exp_name += '_B-{}'.format(config.batch_size)
    if config.step_0:
        exp_name += 'Avg_ece-start-{}'.format(config.ece_train)
        exp_name += '_ece-w{}'.format(config.alpha)
    return exp_name
