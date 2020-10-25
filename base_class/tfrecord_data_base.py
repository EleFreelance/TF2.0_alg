import os
import tensorflow as tf
import numpy as np
from utils.decorate import log_builder
from importlib import import_module
from functools import partial


class tfrecord_data_base(object):
    def __init__(self, config, tfrecord_fn_list=None, preprocess_fn_list=None, tfrecord_parser=None):
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        pass

    def __call__(self, *args, **kwargs):
        pass
