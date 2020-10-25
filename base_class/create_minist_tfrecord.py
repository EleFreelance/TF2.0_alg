import os
import tensorflow as tf
import numpy as np
from utils.decorate import log_builder
from importlib import import_module
from tensorflow.examples.tutorials.mnist import input_data


def _rebuild_tfrecord():
    minist = input_data.read_data_sets('./data', one_hot=True)
    images = minist.train.images
    labels = minist.train.labels
    pixels = images.shape[1]
    num_examples = minist.train.num_examples

    record_file_name = 'record/output.tfrecords'
    if os.path.exists(record_file_name):
        os.remove(record_file_name)

    writer = tf.python_io.TFRecordWriter(record_file_name)

    for i in range(num_examples):
        images_raw = images[i].tostring()
        features = {'images_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images_raw])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.argmax([labels[i]])])),
                    'pixels': tf.train.Feature(int64_list=tf.train.Int64List(value=[pixels]))}
        # 此处只支持关键字参数传递 feature
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    print('data processing success')
    writer.close()


def create_minist_tfrecord(config):
    # 1、检查tfrecord cache是否需要重建
    if config.rebuild_cache:
        # 2、若不存在则根据数据类型生成 tfrecord
        _rebuild_tfrecord()
    # tfrecord文件路径
    record_fn_list = []
    record_file_name = 'record/output.tfrecords'
    record_fn_list.append(record_file_name)

    # 预处理函数
    preprocess_fn_list = []

    # parse tfrecord函数
    package = import_module('run.lib.classifier.%s_parse_func' % config.parse_data_type)
    tfrecord_parser = getattr(package, 'parse_func')

    # 3、若存在则读取tfrecord：并行读取，并行预处理与解析tfrecord，基于prefetch GPU并行训练
    package = import_module('run.lib.classifier.%s_tfrecord_data' % config.parse_data_type)
    tfrecord_reader = getattr(package, 'read_tfrecord_data')

    data_reader = tfrecord_reader(config, record_fn_list, preprocess_fn_list, tfrecord_parser)

    input_data = data_reader()
    # 4、返回batch 数据
    return input_data
