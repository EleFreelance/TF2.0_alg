import os
import tensorflow as tf
import numpy as np
import yaml
from importlib import import_module
import argparse
from easydict import EasyDict

parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument('project', default='E:\\deeplearningCode\\tf_alg', type=str)
# parser.add_argument('output', default='E:\\deeplearningCode\\tf_alg\\output')
parser.add_argument('data_dir', default='E:\\deeplearningCode\\tf_alg\\record')
args = parser.parse_args()


def base_parse():
    config_path = os.path.join(args.project, "base_cfg.yaml")
    with open(config_path) as f:
        config = yaml.load(f.read())
    # easyDict常作为全局参数传入
    config = EasyDict(config)
    return config
