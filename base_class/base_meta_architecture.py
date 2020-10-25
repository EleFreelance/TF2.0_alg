import os
import tensorflow as tf
import numpy as np
from utils.decorate import log_builder
from importlib import import_module
from functools import partial
from abc import ABCMeta, abstractmethod


class base_meta_architecture(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def post_process(self):
        pass

    @abstractmethod
    def info_collection(self, input_data):
        pass

    @abstractmethod
    def predict(self, predict_dict):
        pass

    @abstractmethod
    def eval(self, predict_dict):
        pass

    @abstractmethod
    def loss(self, predict_dict):
        pass

    def restore_map(self, from_checkpoint=True, scope=''):
        # 1、是否从模型的checkpoint中载入变量，分为model与feature_extractor
        if from_checkpoint is False:
            variable_restore_map = {}
            # 2、遍历当前图中所有全局变量，并根据scope对变量进行提取
            for var in tf.global_variables():
                if var.op.name.startswith(self.feature_extractor_scope):
                    variable_restore_map[var.op.name] = var
            return variable_restore_map

        # 3 、若从模型中的checkpoint载入变量，则根据scope与sub_scope进行提取
        variable_restore_map = {}
        if isinstance(scope, (tuple, list)):
            for sub_scope in scope:
                for var in tf.global_variables():
                    if var.op.name.startswith(sub_scope):
                        variable_restore_map[var.op.name] = var
        else:
            for var in tf.global_variables():
                if var.op.name.startswith(scope):
                    variable_restore_map[var.op.name] = var
        return variable_restore_map

    pass

    def feature_extractor(self):
        pass

    @property
    def feature_extractor_scope(self):
        return 'Feature_Extractor'
