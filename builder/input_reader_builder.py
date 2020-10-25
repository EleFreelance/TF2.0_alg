import os
import tensorflow as tf
from glob import glob
from utils.decorate import log_builder
from importlib import import_module


# 使用修饰器对Build函数实现log修饰
@log_builder
def build(config, is_training):
    # 1、get input_reader_config
    input_reader_config = config.input_reader_config

    # 2、cache files check
    cache_files = glob('/record/*.tfrecords')

    # 3、preprocess builder
    preprocess_fn = []

    # 4、 parse func
    package = import_module('run.lib.classifier.%s_parse_func' % config.parse_data_type)
    parse_func = getattr(package, 'parse_func')

    # 5、import tfrecord class
    package = import_module('datasets.%s_%s_tfrecord' % (config.PROJECT_TYPE,input_reader_config.data_type))
    data_reader_class = getattr(package, '%s_%s_tfrecord' % (config.PROJECT_TYPE,input_reader_config.data_type))

    # 6、返回数据读取器对象
    data_reader = data_reader_class(input_reader_config, cache_files, preprocess_fn, is_training,parse_func)
    return data_reader
